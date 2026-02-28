# -*- coding: utf-8 -*-
import hashlib
import json
import pandas as pd
import numpy as np
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from universe import get_master_universe
from data_loader import download_batch, get_ticker_fundamentals, estimate_dividend_yield_from_history
from storage import save_price_data, save_recommendations, initialize_db, save_fundamentals, get_fundamentals, save_model_predictions
from signals import generate_signals
from datetime import datetime
import time
import sys

# Thread pool for parallel processing (optimized for M1 Ultra)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from config import (
    SCANNER_WORKERS,
    PORTFOLIO_HOLDINGS,
    MODEL_PATH,
    CONFLUENCE_THRESHOLD,
    TACTICAL_THRESHOLD,
    TREND_THRESHOLD,
    get_model_thresholds,
    SAFE_ASSET_ATR_THRESHOLD,
    SAFE_ASSET_ALLOWLIST,
    SAFE_ASSET_DENYLIST,
    SAFE_ASSET_BENCHMARK_YIELD_7D,
    SAFE_ASSET_MIN_YIELD_MULTIPLIER,
    SAFE_ASSET_RETURN_DAYS,
    SAFE_ASSET_MIN_RETURN_ABS,
    SAFE_ASSET_TRAILING_DIVIDEND_DAYS,
    SAFE_ASSET_MAX_DIVIDEND_AGE_DAYS,
    SAFE_ASSET_YIELDS_FILE,
    ENABLE_SIGNAL_SMOOTHING,
    SIGNAL_SMOOTHING_ALPHA,
    WATCHLIST,
    UNIVERSE_DENYLIST,
    PORTFOLIO_STALENESS_HOURS,
    MAX_PRICE_DATA_STALENESS_DAYS,
    ALLOW_STALE_DATA_FOR_SIGNALS,
    ENFORCE_TRADABILITY_FILTERS,
    TRADABILITY_MIN_PRICE,
    TRADABILITY_MIN_AVG_VOLUME_20D,
    TRADABILITY_MIN_AVG_DOLLAR_VOLUME_20D,
    TRADABILITY_MAX_ATR_RATIO,
    TRADABILITY_EXEMPT_TICKERS,
    MODEL_DIR,
    MODEL_PATH_5D,
    MODEL_PATH_10D,
    MODEL_PATH_30D,
    HORIZONS,
    HORIZON_TARGETS,
    MODEL_THRESHOLDS_FILE,
    ML_FEATURE_CONTRACT,
)

MAX_WORKERS = SCANNER_WORKERS
SAFE_ASSET_ALLOWSET = {t.upper() for t in SAFE_ASSET_ALLOWLIST}

SAFE_ASSET_DENYSET = {t.upper() for t in SAFE_ASSET_DENYLIST}
UNIVERSE_DENYSET = {t.upper() for t in UNIVERSE_DENYLIST}
TRADABILITY_EXEMPT_SET = {
    t.upper()
    for t in (list(PORTFOLIO_HOLDINGS) + list(WATCHLIST) + list(TRADABILITY_EXEMPT_TICKERS))
    if t
}

STATUS_FILE = os.path.join(PROJECT_ROOT, "SCAN_STATUS.md")
REPORT_FILE = os.path.join(PROJECT_ROOT, "reports", "live_report.md")
REPLAY_VALIDATION_REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
REPLAY_VALIDATION_SAMPLE_KEYS = 25

# Cache for loaded yields and smoothing state
_EXTERNAL_YIELDS_CACHE = None
_SMOOTHING_CACHE = {}  # ticker -> {conf_5d, conf_10d, conf_30d}

# Scan state file for LRU tracking
SCAN_STATE_FILE = os.path.join(PROJECT_ROOT, 'data', 'scan_state.json')


def _load_scan_state() -> dict:
    """Load scan state from JSON file. Returns empty dict on any failure."""
    import json
    try:
        if os.path.exists(SCAN_STATE_FILE):
            with open(SCAN_STATE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"[scan-state] Warning: Could not load {SCAN_STATE_FILE}: {e}")
    return {'last_scanned': {}, 'last_run': {}}


def _save_scan_state(state: dict):
    """Save scan state to JSON file (atomic write)."""
    import json
    temp_file = SCAN_STATE_FILE + '.tmp'
    try:
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=2)
        os.replace(temp_file, SCAN_STATE_FILE)
    except Exception as e:
        print(f"[scan-state] Warning: Could not save scan state: {e}")


def _hours_since(iso_timestamp: str) -> float:
    """Return hours elapsed since the given ISO timestamp."""
    try:
        ts = datetime.fromisoformat(iso_timestamp)
        return (datetime.now() - ts).total_seconds() / 3600
    except:
        return float('inf')  # Treat invalid timestamps as infinitely old


def _normalize_dividend_yield(value: Any) -> float:
    """
    Normalize dividend yield to decimal format (e.g. 4.8% -> 0.048).

    Accepts common malformed inputs like 4.8 or 480 and rescales to decimal.
    """
    try:
        yld = float(value or 0.0)
    except Exception:
        return 0.0
    if not np.isfinite(yld) or yld <= 0:
        return 0.0
    # Rescale percentage-style inputs (e.g. 4.8, 480) into decimal form.
    for _ in range(3):
        if yld <= 1.0:
            break
        yld = yld / 100.0
    return max(0.0, yld)


def _load_external_yields() -> dict:
    """Load external yields from JSON file. Returns empty dict on any failure."""
    global _EXTERNAL_YIELDS_CACHE
    if _EXTERNAL_YIELDS_CACHE is not None:
        return _EXTERNAL_YIELDS_CACHE
    
    try:
        import json
        if os.path.exists(SAFE_ASSET_YIELDS_FILE):
            with open(SAFE_ASSET_YIELDS_FILE, 'r') as f:
                data = json.load(f)
            yields = data.get('yields', {})
            # Normalize keys to uppercase
            _EXTERNAL_YIELDS_CACHE = {
                k.upper(): _normalize_dividend_yield(v)
                for k, v in yields.items()
                if v
            }
            return _EXTERNAL_YIELDS_CACHE
    except Exception as e:
        print(f"  [yields] Warning: Could not load {SAFE_ASSET_YIELDS_FILE}: {e}")
    
    _EXTERNAL_YIELDS_CACHE = {}
    return _EXTERNAL_YIELDS_CACHE


def _get_smoothed_confidence(ticker: str, raw_5d: float, raw_10d: float, raw_30d: float) -> tuple:
    """Apply EMA smoothing to confidence scores if enabled. Returns (smoothed_5d, smoothed_10d, smoothed_30d)."""
    if not ENABLE_SIGNAL_SMOOTHING:
        return raw_5d, raw_10d, raw_30d
    
    alpha = SIGNAL_SMOOTHING_ALPHA
    ticker_u = ticker.upper()
    
    if ticker_u in _SMOOTHING_CACHE:
        prev = _SMOOTHING_CACHE[ticker_u]
        smoothed_5d = alpha * raw_5d + (1 - alpha) * prev.get('conf_5d', raw_5d)
        smoothed_10d = alpha * raw_10d + (1 - alpha) * prev.get('conf_10d', raw_10d)
        smoothed_30d = alpha * raw_30d + (1 - alpha) * prev.get('conf_30d', raw_30d)
    else:
        # First time seeing this ticker - use raw values
        smoothed_5d, smoothed_10d, smoothed_30d = raw_5d, raw_10d, raw_30d
    
    # Update cache for next scan
    _SMOOTHING_CACHE[ticker_u] = {
        'conf_5d': smoothed_5d,
        'conf_10d': smoothed_10d,
        'conf_30d': smoothed_30d,
    }
    
    return smoothed_5d, smoothed_10d, smoothed_30d


def _get_safe_asset_benchmark_yield() -> float:
    """Return benchmark yield for safe-asset gating (fixed 7-day yield)."""
    return float(SAFE_ASSET_BENCHMARK_YIELD_7D or 0.0)


def _get_git_revision_marker() -> str:
    """Return current git revision marker when available."""
    try:
        res = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return (res.stdout or '').strip() or 'no-git'
    except Exception:
        pass
    return 'no-git'


def _sha256_file(path: str) -> str:
    """Return file SHA256 for reproducibility checks."""
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


def _compute_model_version_hash() -> str:
    """
    Hash model artifacts + thresholds + feature contract + revision marker.
    """
    artifact_hashes = {}
    if os.path.isdir(MODEL_DIR):
        for root, _, files in os.walk(MODEL_DIR):
            for name in sorted(files):
                if name.endswith(('.pkl', '.joblib', '.json')):
                    full_path = os.path.join(root, name)
                    rel_path = os.path.relpath(full_path, PROJECT_ROOT)
                    artifact_hashes[rel_path] = _sha256_file(full_path)

    payload = {
        'git_revision': _get_git_revision_marker(),
        'feature_contract_hash': hashlib.sha256(
            json.dumps(ML_FEATURE_CONTRACT, separators=(',', ':')).encode('utf-8')
        ).hexdigest(),
        'thresholds': {
            '5': get_model_thresholds(5),
            '10': get_model_thresholds(10),
            '30': get_model_thresholds(30),
        },
        'confluence': {
            'CONFLUENCE_THRESHOLD': float(CONFLUENCE_THRESHOLD),
            'TACTICAL_THRESHOLD': float(TACTICAL_THRESHOLD),
            'TREND_THRESHOLD': float(TREND_THRESHOLD),
        },
        'core_model_paths': {
            'MODEL_PATH_5D': _sha256_file(MODEL_PATH_5D),
            'MODEL_PATH_10D': _sha256_file(MODEL_PATH_10D),
            'MODEL_PATH_30D': _sha256_file(MODEL_PATH_30D),
            'MODEL_THRESHOLDS_FILE': _sha256_file(MODEL_THRESHOLDS_FILE),
        },
        'artifact_hashes': artifact_hashes,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _compute_data_snapshot_hash(ticker: str, asof_date: pd.Timestamp, slice_df: pd.DataFrame) -> str:
    """Hash as-of input data snapshot for a ticker/date pair."""
    if slice_df.empty:
        return hashlib.sha256(f"{ticker.upper()}|{asof_date.date()}|empty".encode('utf-8')).hexdigest()

    tail = slice_df[['date', 'close', 'volume']].tail(5).copy()
    parts = [ticker.upper(), asof_date.strftime('%Y-%m-%d'), f"rows={len(slice_df)}"]
    for _, row in tail.iterrows():
        d = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
        close_val = float(row.get('close', 0.0) or 0.0)
        vol_raw = row.get('volume', 0)
        try:
            vol_val = int(float(vol_raw))
        except Exception:
            vol_val = 0
        parts.append(f"{d}|{close_val:.8f}|{vol_val}")
    return hashlib.sha256("|".join(parts).encode('utf-8')).hexdigest()


def _coerce_string(value: Any, default: str = "") -> str:
    """Return a normalized string representation for hashing."""
    if value is None:
        return default
    if pd.isna(value):
        return default
    return str(value)


def _coerce_float_for_hash(value: Any, default: float = 0.0) -> float:
    """Return a deterministic float representation for hashing."""
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _coerce_bool_for_hash(value: Any, default: bool = False) -> bool:
    """Return a deterministic bool representation for hashing."""
    if value is None:
        return default
    try:
        if isinstance(value, str):
            val = value.strip().lower()
            if val in ("true", "1", "yes", "y"):
                return True
            if val in ("false", "0", "no", "n"):
                return False
        return bool(value)
    except Exception:
        return default


def _coerce_int_for_hash(value: Any, default: int = 0) -> int:
    """Return a deterministic integer representation for hashing."""
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _coerce_timestamp_for_hash(value: Any, default: str = "") -> str:
    """Return normalized YYYY-MM-DD for date-like values."""
    if value is None or pd.isna(value):
        return default
    try:
        return pd.to_datetime(value, errors='coerce').strftime('%Y-%m-%d')
    except Exception:
        return default


def _replay_row_signature(row: pd.Series) -> Dict[str, Any]:
    """Build canonical signature payload for a single replay prediction row."""
    return {
        'asof_date': _coerce_timestamp_for_hash(row.get('asof_date')),
        'ticker': _coerce_string(row.get('ticker'), default='').upper().strip(),
        'horizon': _coerce_int_for_hash(row.get('horizon')),
        'proba_raw': _coerce_float_for_hash(row.get('proba_raw')),
        'proba_cal': _coerce_float_for_hash(row.get('proba_cal')),
        'score': _coerce_float_for_hash(row.get('score')),
        'signal_type': _coerce_string(row.get('signal_type')),
        'tradable': _coerce_bool_for_hash(row.get('tradable')),
        'tradability_reason': _coerce_string(row.get('tradability_reason')),
        'filters_triggered': _coerce_string(row.get('filters_triggered')),
        'data_snapshot_hash': _coerce_string(row.get('data_snapshot_hash')),
    }


def _replay_row_key(row: pd.Series) -> str:
    """Build a stable lookup key for a replay row."""
    return (
        f"{_coerce_timestamp_for_hash(row.get('asof_date'))}|"
        f"{_coerce_string(row.get('ticker')).upper().strip()}|"
        f"{_coerce_int_for_hash(row.get('horizon'))}"
    )


def _replay_row_hash(row: pd.Series) -> str:
    """Create a deterministic hash for a replay row."""
    signature = _replay_row_signature(row)
    payload = json.dumps(signature, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _snapshot_replay_predictions(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    model_version_hash: str,
) -> Dict[str, Any]:
    """Load and hash replay rows for a deterministic comparison window."""
    from storage import get_connection

    con = get_connection(read_only=True)
    rows = con.execute(
        """
        SELECT asof_date,
               ticker,
               horizon,
               proba_raw,
               proba_cal,
               score,
               signal_type,
               tradable,
               tradability_reason,
               filters_triggered,
               data_snapshot_hash
        FROM model_predictions
        WHERE model_version_hash = ?
          AND asof_date BETWEEN ? AND ?
        ORDER BY asof_date, ticker, horizon
        """,
        [model_version_hash, start_ts.strftime('%Y-%m-%d'), end_ts.strftime('%Y-%m-%d')],
    ).df()
    con.close()

    if rows.empty:
        return {
            'count': 0,
            'row_hash_by_key': {},
            'overall_row_hash': '0' * 64,
            'sample_row_hashes': [],
        }

    rows['asof_date'] = pd.to_datetime(rows['asof_date'], errors='coerce').dt.normalize()
    rows = rows.dropna(subset=['asof_date']).sort_values(['asof_date', 'ticker', 'horizon'])

    row_hash_by_key: Dict[str, str] = {}
    sample_row_hashes: List[str] = []
    for _, row in rows.iterrows():
        key = _replay_row_key(row)
        row_hash = _replay_row_hash(row)
        row_hash_by_key[key] = row_hash
        if len(sample_row_hashes) < REPLAY_VALIDATION_SAMPLE_KEYS:
            sample_row_hashes.append(f"{key}:{row_hash}")

    overall_digest = hashlib.sha256()
    for key in sorted(row_hash_by_key):
        overall_digest.update(f"{key}:{row_hash_by_key[key]}".encode('utf-8'))
    return {
        'count': len(row_hash_by_key),
        'row_hash_by_key': row_hash_by_key,
        'overall_row_hash': overall_digest.hexdigest(),
        'sample_row_hashes': sample_row_hashes,
    }


def _compare_replay_snapshots(
    base: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare two replay snapshots and return detailed mismatch diagnostics."""
    base_rows = base.get('row_hash_by_key', {})
    candidate_rows = candidate.get('row_hash_by_key', {})
    base_keys = set(base_rows)
    candidate_keys = set(candidate_rows)

    missing_in_candidate = sorted(base_keys - candidate_keys)
    missing_in_base = sorted(candidate_keys - base_keys)
    overlap_keys = sorted(base_keys.intersection(candidate_keys))
    mismatched_keys = [
        key for key in overlap_keys
        if base_rows.get(key) != candidate_rows.get(key)
    ]

    mismatch_count = len(missing_in_candidate) + len(missing_in_base) + len(mismatched_keys)
    sample_mismatches = (
        missing_in_candidate[:REPLAY_VALIDATION_SAMPLE_KEYS]
        + missing_in_base[:REPLAY_VALIDATION_SAMPLE_KEYS]
    )
    if len(sample_mismatches) < REPLAY_VALIDATION_SAMPLE_KEYS:
        extra = mismatched_keys[: max(0, REPLAY_VALIDATION_SAMPLE_KEYS - len(sample_mismatches))]
        sample_mismatches.extend(extra)
    sample_mismatches = sample_mismatches[:REPLAY_VALIDATION_SAMPLE_KEYS]

    return {
        'rows_compared': len(overlap_keys),
        'mismatch_count': mismatch_count,
        'missing_in_candidate': missing_in_candidate,
        'missing_in_base': missing_in_base,
        'mismatched_keys': mismatched_keys,
        'sample_keys': sample_mismatches,
        'passes': mismatch_count == 0,
    }


def _write_replay_validation_report(result: Dict[str, Any], report_path: str | None = None) -> str:
    """Write replay reproducibility validation diagnostics and return file path."""
    os.makedirs(REPLAY_VALIDATION_REPORT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not report_path:
        model_tag = _coerce_string(result.get('model_version_hash'), default='no_hash')[:8]
        report_path = os.path.join(
            REPLAY_VALIDATION_REPORT_DIR,
            f"replay_repro_{result.get('start_date', 'start')}_{result.get('end_date', 'end')}_{model_tag}_{timestamp}.md",
        )
    comparison = result.get('comparisons', [])
    last_comparison = comparison[0] if comparison else {}
    sample_keys = last_comparison.get('sample_keys', [])
    mismatched = len(last_comparison.get('mismatched_keys', []))

    with open(report_path, 'w', encoding='utf-8') as report:
        report.write("# Replay Determinism Validation\n\n")
        report.write(f"- Status: {result.get('status')}\n")
        report.write(f"- Start: {result.get('start_date')}\n")
        report.write(f"- End: {result.get('end_date')}\n")
        report.write(f"- Replay runs: {result.get('replay_runs')}\n")
        report.write(f"- Model version hash: `{result.get('model_version_hash')}`\n")
        report.write(f"- Validation status: {'PASS' if not result.get('fatal_error') else 'FAIL'}\n\n")

        report.write("## Diagnostics\n")
        report.write(f"- Rows compared (run1): {result.get('run_summaries', [{}])[0].get('prediction_rows', 0)}\n")
        report.write(f"- Rows compared (run2): {result.get('run_summaries', [{}])[1].get('prediction_rows', 0) if len(result.get('run_summaries', [])) > 1 else 0}\n")
        report.write(f"- Snapshot rows run1: {result.get('snapshots', [{}])[0].get('count', 0)}\n")
        report.write(f"- Snapshot rows run2: {result.get('snapshots', [{},{}])[1].get('count', 0) if len(result.get('snapshots', [])) > 1 else 0}\n")
        report.write(f"- Snapshot overall hash1: {result.get('snapshots', [{}])[0].get('overall_row_hash', '0' * 64)}\n")
        report.write(f"- Snapshot overall hash2: {result.get('snapshots', [{},{}])[1].get('overall_row_hash', '0' * 64) if len(result.get('snapshots', [])) > 1 else '0' * 64}\n")
        report.write(f"- Mismatch rows: {mismatched}\n")
        report.write(f"- Rows compared: {last_comparison.get('rows_compared', 0)}\n")
        report.write(f"- Mismatch keys sample: {sample_keys}\n\n")

        report.write("## Run summaries\n\n")
        for run_index, run_summary in enumerate(result.get('run_summaries', []), start=1):
            report.write(f"- Run {run_index}: rows={run_summary.get('prediction_rows', 0)} hash={run_summary.get('model_version_hash')}\n")

        report.write("\n## Pairwise comparison\n")
        for run_index, comp in enumerate(comparison, start=2):
            report.write(f"- Run 1 vs Run {run_index}: ")
            report.write(f"pass={comp.get('passes')}, ")
            report.write(f"missing_in_run2={len(comp.get('missing_in_candidate', []))}, ")
            report.write(f"missing_in_run1={len(comp.get('missing_in_base', []))}, ")
            report.write(f"mismatched_rows={len(comp.get('mismatched_keys', []))}, ")
            report.write(f"sample_keys={comp.get('sample_keys', [])}\n")

    return report_path


def _is_safe_asset(ticker: str, atr_ratio: float, dividend_yield: float, benchmark_yield: float,
                   return_lookback: float, return_days: int, return_reason: bool = False):
    """Classify safe assets using allow/deny lists and benchmark total-return floors."""
    ticker_u = ticker.upper()
    if ticker_u in SAFE_ASSET_DENYSET:
        return (False, "denylist") if return_reason else False
    if SAFE_ASSET_ALLOWSET and ticker_u not in SAFE_ASSET_ALLOWSET:
        return (False, "not_allowlisted") if return_reason else False

    try:
        atr_val = float(atr_ratio)
        div_val = _normalize_dividend_yield(dividend_yield)
    except (TypeError, ValueError):
        return (False, "bad_numeric") if return_reason else False

    if not np.isfinite(atr_val) or not np.isfinite(div_val):
        return (False, "nan") if return_reason else False
    if atr_val <= 0 or atr_val >= SAFE_ASSET_ATR_THRESHOLD:
        return (False, "atr") if return_reason else False

    min_yield = benchmark_yield * SAFE_ASSET_MIN_YIELD_MULTIPLIER
    if div_val < min_yield:
        return (False, "yield_floor") if return_reason else False
    if return_lookback is None or not np.isfinite(return_lookback):
        return (False, "return_missing") if return_reason else False

    total_return = return_lookback + (div_val * (return_days / 365.0))
    benchmark_total = benchmark_yield * (return_days / 365.0) * SAFE_ASSET_MIN_YIELD_MULTIPLIER
    if total_return < benchmark_total:
        return (False, "below_benchmark") if return_reason else False
    if return_lookback < SAFE_ASSET_MIN_RETURN_ABS:
        return (False, "return_floor") if return_reason else False
    return (True, "ok") if return_reason else True


def _evaluate_tradability(ticker: str, ticker_df: pd.DataFrame, price: float, atr_ratio: float) -> tuple[bool, str, float, float]:
    """Return (tradable, reason, avg_volume_20d, avg_dollar_volume_20d)."""
    ticker_u = ticker.upper()
    if ticker_u in UNIVERSE_DENYSET:
        return False, "denylist", 0.0, 0.0
    if not ENFORCE_TRADABILITY_FILTERS:
        return True, "disabled", 0.0, 0.0
    if ticker_u in TRADABILITY_EXEMPT_SET:
        return True, "exempt", 0.0, 0.0
    if not np.isfinite(price) or price < float(TRADABILITY_MIN_PRICE):
        return False, "price_floor", 0.0, 0.0

    avg_volume_20d = 0.0
    avg_dollar_volume_20d = 0.0
    if 'volume' in ticker_df.columns and 'close' in ticker_df.columns:
        tail = ticker_df[['close', 'volume']].tail(20).copy()
        tail['close'] = pd.to_numeric(tail['close'], errors='coerce')
        tail['volume'] = pd.to_numeric(tail['volume'], errors='coerce')
        tail = tail.dropna(subset=['close', 'volume'])
        if not tail.empty:
            avg_volume_20d = float(tail['volume'].mean())
            avg_dollar_volume_20d = float((tail['close'] * tail['volume']).mean())

    if avg_volume_20d < float(TRADABILITY_MIN_AVG_VOLUME_20D):
        return False, "volume_floor", avg_volume_20d, avg_dollar_volume_20d
    if avg_dollar_volume_20d < float(TRADABILITY_MIN_AVG_DOLLAR_VOLUME_20D):
        return False, "dollar_volume_floor", avg_volume_20d, avg_dollar_volume_20d
    if np.isfinite(atr_ratio) and atr_ratio > float(TRADABILITY_MAX_ATR_RATIO):
        return False, "atr_cap", avg_volume_20d, avg_dollar_volume_20d
    return True, "ok", avg_volume_20d, avg_dollar_volume_20d

def update_live_status(current_chunk: int, total_chunks: int, processed_tickers: int, total_tickers: int, results: list, status_override: str = None, eta_seconds: float = None):
    """Writes real-time status with enhanced technical info to SCAN_STATUS.md."""
    from eta_tracker import format_eta
    
    # Create set for fast lookup (case-insensitive)
    holdings_set = {h.upper() for h in PORTFOLIO_HOLDINGS}
    watchlist_set = {w.upper() for w in WATCHLIST}
    
    # Use memory results
    results_to_display = results if results else []

    # Helper for formatting Term Structure
    def get_term_str(r):
        c5 = r.get('conf_5d', 0.5)
        c10 = r.get('confidence', 0.5)
        c30 = r.get('conf_30d', 0.5)
        def fmt(c):
             # 0 means N/A or not calculated
             if c == 0: return "N/A"
             return f"{'🟢' if c>=0.6 else ('🔴' if c<=0.4 else '➖')} {c:.0%}"
        return f"{fmt(c5)}|{fmt(c10)}|{fmt(c30)}"

    # Categorize Results
    safe_assets = []
    portfolio_signals = []
    watchlist_signals = []
    buy_candidates = []
    holds = []
    sells = []
    
    for r in results_to_display:
        ticker_u = r['ticker'].upper()
        
        # 1. Portfolio
        if ticker_u in holdings_set:
            portfolio_signals.append(r)
            continue 
            
        # 2. Watchlist
        if ticker_u in watchlist_set:
            watchlist_signals.append(r)
            continue 
        
        # 3. Safe Assets
        is_safe = bool(r.get('is_safe_asset')) if r.get('is_safe_asset') is not None else False
        if is_safe:
            safe_assets.append(r)
            continue 
            
        # 4. Main Bucket (Growth/Trading)
        if r['signal_type'] == 'BUY':
            buy_candidates.append(r)
        elif r['signal_type'] == 'SELL':
            sells.append(r)
        else:
            holds.append(r)

    status_text = status_override if status_override else ('Running' if current_chunk < total_chunks else 'Complete')
    
    # Get ML model last modified time
    ml_updated = "Unknown"
    try:
        if os.path.exists(MODEL_PATH):
            ml_mtime = os.path.getmtime(MODEL_PATH)
            ml_updated = datetime.fromtimestamp(ml_mtime).strftime('%Y-%m-%d %H:%M')
    except:
        pass
    
    content = []
    pid = os.getpid()
    content.append(f"# 🚀 Scan Progress Dashboard (PID: {pid})\n")
    content.append(f"**Status**: {status_text} | **Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    content.append(f"**ML Model**: {ml_updated}\n")

    # Progress Bar
    pct = (processed_tickers / total_tickers) if total_tickers else 0
    filled = int(pct * 20)
    bar = "▓" * filled + "░" * (20 - filled)
    eta_str = format_eta(eta_seconds) if eta_seconds else "..."
    content.append(f"### Progress: {pct:.1%} | ETA: **{eta_str}**")
    content.append(f"`{bar}` ({processed_tickers}/{total_tickers})\n")

    # Select Top 15 Buy Opportunities from true BUY signals only.
    top_buys = sorted(buy_candidates, key=lambda x: x.get('confidence', 0), reverse=True)[:15]

    # Stats Summary
    stale_in_results = sum(1 for r in results_to_display if bool(r.get('data_stale')))
    content.append(f"### Statistics")
    content.append(f"- 🟢 Buy Candidates: {len(buy_candidates)} ({len(top_buys)} shown)")
    content.append(f"- ⚪ Holds: {len(holds)}")
    content.append(f"- 🛡️ Safe Assets: {len(safe_assets)}")
    content.append(f"- 💼 Portfolio: {len(portfolio_signals)}")
    content.append(f"- 🕒 Stale Data Used: {stale_in_results}")
    content.append(f"- 🔴 Sells: {len(sells)}\n")
    
    content.append(f"### ℹ️ Signal Legend")
    content.append(f"> * **Term Structure**: 5d|10d|30d ML Confidence (🟢>60%, 🔴<40%).")
    content.append(f"> * **Technical**: RSI (Relative Strength Index) | Trend against SMA200 (↑/⬇️).")
    content.append(f"> * **Safe Assets**: Allowlist + ATR<{SAFE_ASSET_ATR_THRESHOLD:.1%} + Yield >= {SAFE_ASSET_BENCHMARK_YIELD_7D:.2%} (7d benchmark) + {SAFE_ASSET_RETURN_DAYS}d Return >= {SAFE_ASSET_MIN_RETURN_ABS:.2%}.")
    content.append(f"> * **Div**: 💰 indicates dividend paying stock.\n")

    # Atomic write setup
    temp_file = STATUS_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(content))
        
        # 1. Portfolio Holdings (Conf DESC) - MOVED TO TOP
        if portfolio_signals:
            f.write(f"### 💼 Portfolio Holdings Status\n")
            # Removed redundant 'Signal' column, merged into Term Structure/Action implied
            f.write("| Ticker | Price | Term Structure (5d|10d|30d) | Technicals | Div |\n")
            f.write("|--------|-------|-----------------------------|------------|-----|\n")
            for p in sorted(portfolio_signals, key=lambda x: x.get('confidence', 0), reverse=True):
                 rsi_val = f"{p['rsi']:.0f}"
                 trend_sym = '↑' if p['price_at_rec'] > p['sma_200'] else '⬇'
                 tech_summary = f"RSI:{rsi_val} {trend_sym} SMA"
                 div = '💰' if p.get('is_dividend') else ''
                 f.write(f"| {p['ticker']} | {p['price_at_rec']:.2f} | {get_term_str(p)} | {tech_summary} | {div} |\n")
            f.write("\n")

        # 2. Buy Opportunities (Top 15)
        if top_buys:
            f.write(f"### 🟢 Top 15 Buy Opportunities (Best ML Signals)\n")
            f.write("| Ticker | Price | Term Structure (5d|10d|30d) | Technicals | Div |\n")
            f.write("|--------|-------|-----------------------------|------------|-----|\n")
            for b in top_buys:
                div = '💰' if b.get('is_dividend') else ''
                rsi_val = f"{b['rsi']:.0f}"
                trend_sym = '↑' if b['price_at_rec'] > b['sma_200'] else '⬇'
                tech_summary = f"RSI:{rsi_val} {trend_sym} SMA"
                f.write(f"| {b['ticker']} | {b['price_at_rec']:.2f} | {get_term_str(b)} | {tech_summary} | {div} |\n")
            f.write("\n")

        # 3. Watchlist (Conf DESC)
        if watchlist_signals:
            f.write(f"### 👀 Watchlist Status\n")
            f.write("| Ticker | Price | Term Structure (5d|10d|30d) | Div |\n")
            f.write("|--------|-------|-----------------------------|-----|\n")
            for w in sorted(watchlist_signals, key=lambda x: x.get('confidence', 0), reverse=True):
                div = '💰' if w.get('is_dividend') else ''
                f.write(f"| {w['ticker']} | {w['price_at_rec']:.2f} | {get_term_str(w)} | {div} |\n")
            f.write("\n")

        # 4. Safe Assets (Yield DESC)
        if safe_assets:
            f.write(f"### 🛡️ Top Safe Assets (Yield/Income)\n")
            f.write(f"> Benchmark 7d Yield: {SAFE_ASSET_BENCHMARK_YIELD_7D:.2%}\n")
            # Columns: Yield, 30d Return, Spread (yield vs benchmark), Vol, Safety Rating
            f.write("| Ticker | Price | Yield | 30d Ret | Spread | Vol | Safety |\n")
            f.write("|--------|-------|-------|---------|--------|-----|--------|\n")
            for sa in sorted(safe_assets, key=lambda x: x.get('dividend_yield', 0), reverse=True)[:15]:
                yld = _normalize_dividend_yield(sa.get('dividend_yield', 0) or 0)
                yld_val = f"{yld*100:.2f}%"
                ret_30d = sa.get('returns_30d')
                ret_val = f"{ret_30d:.2%}" if isinstance(ret_30d, (int, float)) else "N/A"
                # Yield spread vs benchmark
                spread = yld - SAFE_ASSET_BENCHMARK_YIELD_7D
                spread_val = f"+{spread*100:.1f}%" if spread >= 0 else f"{spread*100:.1f}%"
                vol = sa.get('atr_ratio', 0) or 0
                vol_val = f"{vol:.2%}"
                # Safety rating based on volatility: ⭐⭐⭐ (<0.05%), ⭐⭐ (<0.15%), ⭐ (<0.25%)
                if vol < 0.0005:
                    safety = "⭐⭐⭐"
                elif vol < 0.0015:
                    safety = "⭐⭐"
                elif vol < 0.0025:
                    safety = "⭐"
                else:
                    safety = "⚠️"
                price = sa.get('price_at_rec')
                price_val = f"{price:.2f}" if isinstance(price, (int, float)) else "N/A"
                f.write(f"| {sa['ticker']} | {price_val} | {yld_val} | {ret_val} | {spread_val} | {vol_val} | {safety} |\n")
            f.write("\n")

        # 5. Sell Signals (Conf ASC, Limit 15)
        if sells:
            f.write(f"### 🔴 Latest SELL Signals\n")
            f.write("| Ticker | Price | Term Structure (5d|10d|30d) | Technicals |\n")
            f.write("|--------|-------|-----------------------------|------------|\n")
            for s in sorted(sells, key=lambda x: x.get('confidence', 1.0), reverse=False)[:15]:
                 rsi_val = f"{s['rsi']:.0f}"
                 trend_sym = '↑' if s['price_at_rec'] > s['sma_200'] else '⬇'
                 tech_summary = f"RSI:{rsi_val} {trend_sym} SMA"
                 f.write(f"| {s['ticker']} | {s['price_at_rec']:.2f} | {get_term_str(s)} | {tech_summary} |\n")
            f.write("\n")

    os.replace(temp_file, STATUS_FILE)

def run_full_scan(chunk_size: int = 20, return_meta: bool = False):
    """
    Runs a full market scan for all tickers in the universe.
    Prevents concurrent runs using a PID file.
    """
    from eta_tracker import ETATracker
    from safe_logger import start_logging, stop_logging
    import os
    import sys
    import atexit

    # --- PID Lock Mechanism ---
    PID_FILE = os.path.join(PROJECT_ROOT, 'market_scan.pid')
    
    def remove_pid_file():
        if os.path.exists(PID_FILE):
            try:
                os.remove(PID_FILE)
            except:
                pass

    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                old_pid = int(f.read().strip())
            try:
                os.kill(old_pid, 0)
                print(f"[STOP] Scan already running (PID: {old_pid}). Exiting.")
                if return_meta:
                    return [], {}
                return []
            except OSError:
                print(f"[WARNING] Found stale PID file ({old_pid}). Overwriting.")
        except ValueError:
            print("[WARNING] Invalid PID file. Overwriting.")
    
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    
    atexit.register(remove_pid_file)
    
    start_logging()
    initialize_db()
    
    # --- LRU Scan Recovery Logic ---
    scan_state = _load_scan_state()
    last_run = scan_state.get('last_run', {})
    
    if last_run.get('status') in ['interrupted', 'failed']:
        from universe import get_lru_universe
        
        # Check if portfolio is stale (>PORTFOLIO_STALENESS_HOURS old)
        last_scanned = scan_state.get('last_scanned', {})
        portfolio_timestamps = [last_scanned.get(t.upper()) for t in PORTFOLIO_HOLDINGS]
        valid_timestamps = [t for t in portfolio_timestamps if t]
        
        if valid_timestamps:
            oldest_portfolio_hours = max(_hours_since(t) for t in valid_timestamps)
        else:
            oldest_portfolio_hours = float('inf')  # Never scanned
        
        if oldest_portfolio_hours < PORTFOLIO_STALENESS_HOURS:
            # Portfolio is fresh - pure LRU ordering
            print(f"[scan-state] Last run {last_run.get('status')}. Portfolio scanned {oldest_portfolio_hours:.1f}h ago - using pure LRU.", flush=True)
            universe = get_lru_universe(prioritize_portfolio=False)
        else:
            # Portfolio is stale - prioritize it, then LRU
            print(f"[scan-state] Last run {last_run.get('status')}. Portfolio stale ({oldest_portfolio_hours:.1f}h) - prioritizing portfolio, then LRU.", flush=True)
            universe = get_lru_universe(prioritize_portfolio=True)
    else:
        # Normal run - standard priority order
        universe = get_master_universe()

    if UNIVERSE_DENYSET:
        before = len(universe)
        universe = [t for t in universe if t.upper() not in UNIVERSE_DENYSET]
        removed = before - len(universe)
        if removed > 0:
            print(f"[universe] Excluded {removed} denylisted ticker(s): {sorted(UNIVERSE_DENYSET)}", flush=True)
    
    total_tickers = len(universe)
    total_chunks = (total_tickers - 1) // chunk_size + 1
    scan_start_time = datetime.now()
    print(f"Starting full market scan for {total_tickers} tickers...", flush=True)

    tracker = ETATracker("market_scan")
    tracker.start_run({"tickers": total_tickers, "chunk_size": chunk_size})
    
    try:
        from macro_loader import get_macro_features
        macro_data = get_macro_features()
        print(f"Loaded {len(macro_data)} macro features for ML inference.", flush=True)
    except Exception as e:
        print(f"Warning: Could not load macro data: {e}. Using defaults.", flush=True)
        macro_data = {}

    try:
        from ml_engine import load_qual_features
        qual_cache = load_qual_features()
        if qual_cache:
            print(f"Loaded qual features for {len(qual_cache)} tickers for ML inference.", flush=True)
    except Exception as e:
        print(f"Warning: Could not load qual features: {e}. Using defaults.", flush=True)
        qual_cache = {}

    try:
        from news_loader import get_news_features
        news_data = get_news_features()
        print(f"Loaded news features: {news_data}", flush=True)
    except Exception as e:
        print(f"Warning: Could not load news features: {e}. Using defaults.", flush=True)
        news_data = {}
    
    # Auto-refresh external yields (ASYNC - runs in background, doesn't block scan)
    try:
        import subprocess
        scripts_dir = os.path.join(PROJECT_ROOT, 'scripts')
        update_script = os.path.join(scripts_dir, 'update_yields.py')
        if os.path.exists(update_script):
            print("[yields] Starting async yields update in background...", flush=True)
            # Non-blocking: start process and continue immediately
            # Output goes to devnull to avoid blocking on pipe buffers
            devnull = open(os.devnull, 'w')
            subprocess.Popen(
                ['python3', update_script],
                cwd=PROJECT_ROOT,
                stdout=devnull,
                stderr=devnull,
                start_new_session=True,  # Detach from parent process
            )
            print("[yields] Background update started. Scan continues immediately.", flush=True)
    except Exception as e:
        print(f"[yields] Could not start yields update: {e}. Continuing with cached data.", flush=True)
    
    # Clear cached yields to reload fresh data (will pick up results when script finishes)
    global _EXTERNAL_YIELDS_CACHE
    _EXTERNAL_YIELDS_CACHE = None
    
    benchmark_yield = _get_safe_asset_benchmark_yield()
    thresholds_5d = get_model_thresholds(5)
    thresholds_10d = get_model_thresholds(10)
    thresholds_30d = get_model_thresholds(30)
    print(
        "Loaded model thresholds: "
        f"5d BUY>={thresholds_5d.get('buy', 0.60):.2f}/SELL<={thresholds_5d.get('sell', 0.40):.2f}, "
        f"10d BUY>={thresholds_10d.get('buy', 0.60):.2f}/SELL<={thresholds_10d.get('sell', 0.40):.2f}, "
        f"30d BUY>={thresholds_30d.get('buy', 0.60):.2f}/SELL<={thresholds_30d.get('sell', 0.40):.2f}",
        flush=True,
    )
    update_live_status(0, total_chunks, 0, total_tickers, [], status_override="Initializing Data Sources...")
    all_results = []
    stale_tickers = set()
    failed_fetch_tickers = set()
    insufficient_history_tickers = set()

    def _current_scan_meta() -> dict:
        return {
            'total_tickers': total_tickers,
            'results_generated': len(all_results),
            'stale_tickers': len(stale_tickers),
            'failed_fetch_tickers': len(failed_fetch_tickers),
            'insufficient_history_tickers': len(insufficient_history_tickers),
        }
    
    try:
        for i in range(0, total_tickers, chunk_size):
            chunk = universe[i:i + chunk_size]
            current_chunk_idx = i//chunk_size + 1
            
            progress = i / total_tickers
            remaining_eta = tracker.get_live_eta(progress)
            eta_str = f" | ETA: {remaining_eta/60:.1f}min" if remaining_eta else ""
            print(f"Processing chunk {current_chunk_idx}/{total_chunks} ({len(chunk)} tickers)...{eta_str}", flush=True)
            
            from storage import get_connection
            con = get_connection()
            today = datetime.now().strftime('%Y-%m-%d')
            existing_count = con.execute("SELECT COUNT(*) FROM price_history WHERE ticker IN (SELECT * FROM (SELECT unnest(?))) AND date = ?", [chunk, today]).fetchone()[0]
            con.close()
            
            chunk_from_cache = existing_count >= len(chunk) * 0.8
            if chunk_from_cache:
                print(f"Skipping chunk {current_chunk_idx} as data is already cached.")
                df_batch = pd.DataFrame()
                for ticker in chunk:
                    from storage import load_price_data
                    df_ticker = load_price_data(ticker)
                    df_batch = pd.concat([df_batch, df_ticker])
            else:
                from data_loader import download_batch_with_fallback
                max_retries = 3
                retry_delay = 15
                for attempt in range(max_retries):
                    df_batch = download_batch_with_fallback(chunk)
                    if not df_batch.empty:
                        break
                    if attempt < max_retries - 1:
                        print(f"  [Chunk {current_chunk_idx}] Failed. Retrying in {retry_delay}s... ({attempt+1}/{max_retries})", flush=True)
                        time.sleep(retry_delay)
                
                if not df_batch.empty:
                    save_price_data(df_batch)
                else:
                    print(f"  CRITICAL: All download methods failed for chunk {current_chunk_idx} after {max_retries} attempts.")
            
            processed_chunk_results = []
            for ticker in chunk:
                try:
                    ticker_df = pd.DataFrame()
                    data_source = "download_batch"
                    if not df_batch.empty:
                        ticker_df = df_batch[df_batch['ticker'] == ticker].sort_values('date')
                        if chunk_from_cache:
                            data_source = "cache_chunk"
                    
                    if ticker_df.empty:
                        from storage import load_price_data
                        ticker_df = load_price_data(ticker)
                        data_source = "cache_fallback"
                        if not ticker_df.empty:
                            stale_date = ticker_df['date'].max()
                            if stale_date is not None:
                                try:
                                    stale_str = stale_date.strftime('%Y-%m-%d')
                                except Exception:
                                    stale_str = str(stale_date)
                                print(f"  Warning: Using STALE data for {ticker} (latest: {stale_str})")
                            else:
                                print(f"  Warning: Using STALE data for {ticker} (latest date unknown)")
                    
                    if ticker_df.empty:
                        failed_fetch_tickers.add(ticker.upper())
                        continue

                    max_dt = pd.to_datetime(ticker_df['date'].max())
                    age_days = int((pd.Timestamp(datetime.now().date()) - max_dt.normalize()).days)
                    is_stale = age_days > MAX_PRICE_DATA_STALENESS_DAYS
                    if is_stale:
                        stale_tickers.add(ticker.upper())
                        if not ALLOW_STALE_DATA_FOR_SIGNALS:
                            failed_fetch_tickers.add(ticker.upper())
                            print(
                                f"  Warning: Skipping {ticker} due to stale data ({age_days}d old, max {MAX_PRICE_DATA_STALENESS_DAYS}d)",
                                flush=True,
                            )
                            continue
                    
                    if len(ticker_df) < 200:
                        insufficient_history_tickers.add(ticker.upper())
                        continue
                        
                    ticker_df = generate_signals(ticker_df, ticker=ticker)
                    last_row = ticker_df.iloc[-1]
                    returns_lookback = None
                    if len(ticker_df) > SAFE_ASSET_RETURN_DAYS:
                        prev_close = float(ticker_df['close'].iloc[-1 - SAFE_ASSET_RETURN_DAYS])
                        if prev_close > 0:
                            returns_lookback = (float(last_row['close']) / prev_close) - 1.0
                    
                    # --- NEW: Multi-API Fundamentals Fetching ---
                    # Check DB Cache first
                    fund_data = get_fundamentals(ticker)
                    
                    # Refresh if missing or stale (simple check: if no yield data)
                    if not fund_data or not fund_data.get('dividend_yield'):
                         fund_data = get_ticker_fundamentals(ticker)
                         if fund_data.get('source') != 'none':
                             save_fundamentals(ticker, fund_data)
                    
                    raw_db_yield = fund_data.get('dividend_yield', 0)
                    div_yield = _normalize_dividend_yield(raw_db_yield)
                    try:
                        raw_db_float = float(raw_db_yield)
                    except Exception:
                        raw_db_float = 0.0
                    if raw_db_float > 0 and abs(div_yield - raw_db_float) > 1e-9:
                        fund_data['dividend_yield'] = div_yield
                        save_fundamentals(ticker, fund_data)
                    if ticker.upper() in SAFE_ASSET_ALLOWSET:
                        hist_yield = estimate_dividend_yield_from_history(
                            ticker,
                            SAFE_ASSET_TRAILING_DIVIDEND_DAYS,
                            SAFE_ASSET_MAX_DIVIDEND_AGE_DAYS,
                        )
                        hist_yield = _normalize_dividend_yield(hist_yield)
                        if hist_yield and hist_yield > div_yield:
                            div_yield = hist_yield
                            fund_data['dividend_yield'] = div_yield
                            save_fundamentals(ticker, fund_data)
                    
                    # Fallback to external yields file if API failed
                    if div_yield <= 0 and ticker.upper() in SAFE_ASSET_ALLOWSET:
                        ext_yields = _load_external_yields()
                        ext_yield = _normalize_dividend_yield(ext_yields.get(ticker.upper(), 0))
                        if ext_yield > 0:
                            print(f"  [yields] Using external yield {ext_yield:.2%} for {ticker}")
                            div_yield = ext_yield
                    
                    is_dividend = div_yield > 0.01
                    div_info = {'dividend_yield': div_yield, 'is_dividend': is_dividend}
                    
                    # --------------------------------------------
                    
                    rec_status = "HOLD"
                    reason = "ML Neutral"
                    
                    from ml_engine import extract_ml_features, get_multi_horizon_confidence
                    qual_data = qual_cache.get(ticker.upper()) if qual_cache else None
                    ticker_feat = extract_ml_features(
                        ticker_df,
                        dividend_info=div_info,
                        macro_data=macro_data,
                        qual_data=qual_data,
                        news_data=news_data,
                        horizon=10,
                        for_inference=True,
                    )
                    
                    if ticker_feat.empty:
                        conf = 0.5
                        conf_5d = 0.5
                        conf_30d = 0.5
                        raw_conf = 0.5
                        raw_5d = 0.5
                        raw_30d = 0.5
                    else:
                        last_row_feat = ticker_feat.iloc[-1]
                        horizon_confs = get_multi_horizon_confidence(last_row_feat, ticker=ticker)
                        conf = horizon_confs.get('conf_10d', 0.5)
                        conf_5d = horizon_confs.get('conf_5d', 0.5)
                        conf_30d = horizon_confs.get('conf_30d', 0.5)
                        raw_conf = horizon_confs.get('raw_conf_10d', conf)
                        raw_5d = horizon_confs.get('raw_conf_5d', conf_5d)
                        raw_30d = horizon_confs.get('raw_conf_30d', conf_30d)
                        # Apply EMA smoothing if enabled
                        conf_5d, conf, conf_30d = _get_smoothed_confidence(ticker, raw_5d, raw_conf, raw_30d)
                    
                    buy_thr_5d = float(thresholds_5d.get('buy', 0.60))
                    sell_thr_5d = float(thresholds_5d.get('sell', 0.40))
                    buy_thr_10d = float(thresholds_10d.get('buy', 0.60))
                    sell_thr_10d = float(thresholds_10d.get('sell', 0.40))
                    buy_thr_30d = float(thresholds_30d.get('buy', 0.60))
                    sell_thr_30d = float(thresholds_30d.get('sell', 0.40))

                    strong_buy = (
                        conf_5d >= max(buy_thr_5d, CONFLUENCE_THRESHOLD)
                        and conf >= max(buy_thr_10d, CONFLUENCE_THRESHOLD)
                        and conf_30d >= max(buy_thr_30d, CONFLUENCE_THRESHOLD)
                    )
                    tactical_buy = (
                        conf_5d >= max(buy_thr_5d, TACTICAL_THRESHOLD)
                        and conf >= buy_thr_10d
                    )
                    trend_buy = (
                        conf_30d >= max(buy_thr_30d, TREND_THRESHOLD)
                        and conf >= buy_thr_10d
                    )
                    confirmed_sell = (
                        conf <= sell_thr_10d
                        and (conf_5d <= sell_thr_5d or conf_30d <= sell_thr_30d)
                    )

                    if strong_buy:
                        rec_status = "BUY"
                        reason = f"Strong Confluence ({conf_5d:.0%}|{conf:.0%}|{conf_30d:.0%})"
                    elif tactical_buy:
                        rec_status = "BUY"
                        reason = f"Tactical Confluence ({conf_5d:.0%}|{conf:.0%}|{conf_30d:.0%})"
                    elif trend_buy:
                        rec_status = "BUY"
                        reason = f"Trend Confluence ({conf_5d:.0%}|{conf:.0%}|{conf_30d:.0%})"
                    elif confirmed_sell:
                        rec_status = "SELL"
                        reason = f"Multi-Horizon Weakness ({conf_5d:.0%}|{conf:.0%}|{conf_30d:.0%})"
                    else:
                        reason = f"Neutral/No Confluence ({conf_5d:.0%}|{conf:.0%}|{conf_30d:.0%})"
                    
                    if rec_status == "BUY":
                        if last_row['signal'] == 1: reason += ", Trend"
                    elif rec_status == "SELL":
                         if last_row['signal'] == 0: reason += ", Tech Sell"

                    macd_status = '📈' if last_row.get('macd_bullish', 0) == 1 else '📉'
                    price_at_rec = float(last_row['close'])
                    atr_ratio_val = float(last_row.get('atr_ratio', 0) or 0)
                    if atr_ratio_val == 0 and not ticker_feat.empty:
                        atr_ratio_val = float(ticker_feat.iloc[-1].get('atr_ratio', 0) or 0)

                    tradable, tradability_reason, avg_volume_20d, avg_dollar_volume_20d = _evaluate_tradability(
                        ticker=ticker,
                        ticker_df=ticker_df,
                        price=price_at_rec,
                        atr_ratio=atr_ratio_val,
                    )
                    if rec_status == "BUY" and not tradable:
                        rec_status = "HOLD"
                        reason = f"Filtered BUY ({tradability_reason}) | {reason}"

                    res = {
                        'date': last_row['date'].strftime('%Y-%m-%d'),
                        'ticker': ticker,
                        'signal_type': rec_status,
                        'price_at_rec': price_at_rec,
                        'rsi': float(last_row['rsi']),
                        'sma_50': float(last_row['sma_50']),
                        'sma_200': float(last_row['sma_200']),
                        'confidence': float(conf),
                        'raw_confidence': float(raw_conf),
                        'conf_5d': float(conf_5d),
                        'raw_conf_5d': float(raw_5d),
                        'conf_30d': float(conf_30d),
                        'raw_conf_30d': float(raw_30d),
                        'macd_status': macd_status,
                        'dividend_yield': div_yield,
                        'returns_30d': returns_lookback,
                        'days_to_ex_div': None, # Removed hard dependency
                        'is_dividend': is_dividend,
                        'status': 'OPEN',
                        'reason': reason,
                        'tradable': bool(tradable),
                        'tradability_reason': tradability_reason,
                        'avg_volume_20d': float(avg_volume_20d),
                        'avg_dollar_volume_20d': float(avg_dollar_volume_20d),
                        'data_stale': bool(is_stale),
                        'data_age_days': int(max(age_days, 0)),
                        'data_source': data_source,
                    }

                    res['atr_ratio'] = atr_ratio_val
                    # Pass extra fundamental fields
                    res['sector'] = fund_data.get('sector', '')
                    res['pe_ratio'] = fund_data.get('pe_ratio', 0)
                    
                    is_safe, safe_reason = _is_safe_asset(
                        ticker,
                        res['atr_ratio'],
                        div_yield,
                        benchmark_yield,
                        returns_lookback,
                        SAFE_ASSET_RETURN_DAYS,
                        return_reason=True,
                    )
                    res['is_safe_asset'] = is_safe
                    if (ticker.upper() in SAFE_ASSET_ALLOWSET) and not is_safe:
                        print(f"  [safe-asset] {ticker} rejected: {safe_reason}", flush=True)
                    
                    processed_chunk_results.append(res)
                    all_results.append(res)
                    term_str = f"{conf_5d:.0%}|{conf:.0%}|{conf_30d:.0%}"
                    print(f"  - {ticker}: {rec_status} (Term: {term_str})", flush=True)

                except Exception as e:
                    print(f"Error processing {ticker}: {e}", flush=True)
                
            if processed_chunk_results:
                save_recommendations(processed_chunk_results, scan_time=scan_start_time)
                
                # Update scan state with timestamps for processed tickers
                for res in processed_chunk_results:
                    scan_state['last_scanned'][res['ticker'].upper()] = datetime.now().isoformat()
                # Save state every chunk to persist progress
                scan_state['last_run'] = {
                    'status': 'running',
                    'timestamp': datetime.now().isoformat(),
                    'tickers_completed': i + len(chunk),
                    'total_tickers': total_tickers
                }
                _save_scan_state(scan_state)
                
            if current_chunk_idx == 1 or current_chunk_idx % 2 == 0 or current_chunk_idx == total_chunks:
                update_live_status(current_chunk_idx, total_chunks, i + len(chunk), total_tickers, all_results, eta_seconds=remaining_eta)
            
            if current_chunk_idx % 2 == 0 or current_chunk_idx == total_chunks:
                from reporter import generate_market_report
                scan_meta = _current_scan_meta()
                generate_market_report(all_results, scan_meta=scan_meta)
                print(f"  📝 Incremental report generated ({current_chunk_idx}/{total_chunks})")
                
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nScan interrupted by user.")
        update_live_status(locals().get('current_chunk_idx',0), total_chunks, locals().get('i',0), total_tickers, all_results, status_override="⚠️ Interrupted (User)")
        # Save interrupted state for LRU recovery
        scan_state['last_run'] = {
            'status': 'interrupted',
            'timestamp': datetime.now().isoformat(),
            'tickers_completed': locals().get('i', 0),
            'total_tickers': total_tickers
        }
        _save_scan_state(scan_state)
        tracker.end_run()
        raise
    except Exception as e:
        print(f"\nScan failed: {e}")
        update_live_status(locals().get('current_chunk_idx',0), total_chunks, locals().get('i',0), total_tickers, all_results, status_override=f"❌ Failed: {str(e)[:50]}...")
        # Save failed state for LRU recovery
        scan_state['last_run'] = {
            'status': 'failed',
            'timestamp': datetime.now().isoformat(),
            'tickers_completed': locals().get('i', 0),
            'total_tickers': total_tickers,
            'error': str(e)[:100]
        }
        _save_scan_state(scan_state)
        tracker.end_run()
        raise
    finally:
        if locals().get('current_chunk_idx') == total_chunks:
            update_live_status(total_chunks, total_chunks, total_tickers, total_tickers, all_results, status_override="✅ Complete", eta_seconds=0)
            # Save completed state
            scan_state['last_run'] = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'tickers_completed': total_tickers,
                'total_tickers': total_tickers
            }
            _save_scan_state(scan_state)

    tracker.end_run()
    
    try:
        from data_loader import get_api_stats
        stats = get_api_stats()
        print("\n" + "="*60, flush=True)
        print("API PERFORMANCE SUMMARY", flush=True)
        print("="*60, flush=True)
        for api, data in stats.items():
            if data['calls'] > 0:
                avg_time = data['total_time'] / data['calls']
                success_pct = (data['success'] / data['calls']) * 100
                print(f"  {api:15} | Calls: {data['calls']:4} | Success: {success_pct:5.1f}% | Avg: {avg_time:.2f}s", flush=True)
        print("="*60, flush=True)
    except Exception as e:
        print(f"Could not print API stats: {e}", flush=True)
    
    stop_logging()
    scan_meta = _current_scan_meta()
    print(
        "SCAN_META "
        f"total_tickers={scan_meta.get('total_tickers', 0)} "
        f"results_generated={scan_meta.get('results_generated', 0)} "
        f"stale_tickers={scan_meta.get('stale_tickers', 0)} "
        f"failed_fetch_tickers={scan_meta.get('failed_fetch_tickers', 0)} "
        f"insufficient_history_tickers={scan_meta.get('insufficient_history_tickers', 0)}",
        flush=True,
    )
    print("Scan complete.", flush=True)
    if return_meta:
        return all_results, scan_meta
    return all_results


def _load_replay_dividend_context(
    tickers: List[str],
    end_ts: pd.Timestamp,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Any]]]:
    """
    Load best-effort historical dividend context for replay inference.

    Priority source: recommendation_history dividend_yield snapshots by date.
    Fallback source: fundamentals snapshot only if updated_at <= as-of date.
    """
    history_cache: Dict[str, Dict[str, np.ndarray]] = {}
    fundamentals_cache: Dict[str, Dict[str, Any]] = {}
    if not tickers:
        return history_cache, fundamentals_cache

    from storage import get_connection
    con = get_connection(read_only=True)
    try:
        hist_df = con.execute(
            """
            SELECT upper(ticker) AS ticker, date, dividend_yield
            FROM recommendation_history
            WHERE upper(ticker) IN (SELECT * FROM (SELECT UNNEST(?)))
              AND date <= ?
              AND dividend_yield IS NOT NULL
            ORDER BY ticker, date
            """,
            [tickers, end_ts.strftime('%Y-%m-%d')],
        ).df()
    except Exception:
        hist_df = pd.DataFrame()

    try:
        fund_df = con.execute(
            """
            SELECT upper(ticker) AS ticker, dividend_yield, updated_at
            FROM fundamentals
            WHERE upper(ticker) IN (SELECT * FROM (SELECT UNNEST(?)))
            """,
            [tickers],
        ).df()
    except Exception:
        fund_df = pd.DataFrame()
    finally:
        con.close()

    if not hist_df.empty:
        hist_df['date'] = pd.to_datetime(hist_df['date'], errors='coerce').dt.normalize()
        hist_df['dividend_yield'] = pd.to_numeric(hist_df['dividend_yield'], errors='coerce')
        hist_df['dividend_yield'] = hist_df['dividend_yield'].apply(_normalize_dividend_yield)
        hist_df = hist_df.dropna(subset=['date', 'ticker']).sort_values(['ticker', 'date'])
        for ticker, grp in hist_df.groupby('ticker', sort=False):
            dates = grp['date'].to_numpy(dtype='datetime64[ns]')
            yields = grp['dividend_yield'].fillna(0.0).to_numpy(dtype=float)
            if len(dates) > 0:
                history_cache[str(ticker).upper()] = {'dates': dates, 'yields': yields}

    if not fund_df.empty:
        fund_df['updated_at'] = pd.to_datetime(fund_df['updated_at'], errors='coerce')
        for _, row in fund_df.iterrows():
            ticker = str(row.get('ticker', '')).upper().strip()
            if not ticker:
                continue
            try:
                div_yield = _normalize_dividend_yield(row.get('dividend_yield', 0) or 0)
            except Exception:
                div_yield = 0.0
            fundamentals_cache[ticker] = {
                'dividend_yield': div_yield,
                'updated_at': row.get('updated_at'),
            }

    return history_cache, fundamentals_cache


def _resolve_replay_dividend_info(
    ticker: str,
    asof_date: pd.Timestamp,
    asof_np: np.datetime64,
    history_cache: Dict[str, Dict[str, np.ndarray]],
    fundamentals_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve best-effort dividend context without future timestamp leakage."""
    ticker_u = str(ticker).upper()
    div_yield = 0.0

    hist_ctx = history_cache.get(ticker_u)
    if hist_ctx:
        dates = hist_ctx.get('dates')
        yields = hist_ctx.get('yields')
        if dates is not None and yields is not None and len(dates) > 0:
            pos = int(np.searchsorted(dates, asof_np, side='right')) - 1
            if pos >= 0 and pos < len(yields):
                try:
                    div_yield = _normalize_dividend_yield(yields[pos] or 0.0)
                except Exception:
                    div_yield = 0.0

    if div_yield <= 0:
        fund_ctx = fundamentals_cache.get(ticker_u)
        if fund_ctx:
            updated_at = pd.to_datetime(fund_ctx.get('updated_at'), errors='coerce')
            if pd.notna(updated_at) and updated_at.normalize() <= asof_date:
                try:
                    div_yield = _normalize_dividend_yield(fund_ctx.get('dividend_yield', 0) or 0.0)
                except Exception:
                    div_yield = 0.0

    return {
        'dividend_yield': max(0.0, div_yield),
        'dividend_rate': 0.0,
        'days_to_ex_div': None,
        'payout_ratio': 0.0,
        'is_dividend': bool(div_yield > 0.01),
        'dividend_safety_score': 0.5,
        'is_yield_trap': False,
        'dividend_growth_5y': 0.0,
    }


def _apply_replay_context_to_feature_row(
    feature_row: pd.Series,
    macro_data: Dict[str, float],
    dividend_info: Dict[str, Any],
    news_data: Dict[str, float] = None,
) -> pd.Series:
    """Inject as-of macro, news, and dividend context into a precomputed feature row."""
    row = feature_row.copy()
    macro = macro_data or {}

    macro_vix = float(macro.get('macro_VIXCLS', 20) or 20)
    macro_yield_curve = float(macro.get('macro_T10Y2Y', 0) or 0)
    row['macro_vix'] = macro_vix
    row['macro_yield_curve'] = macro_yield_curve
    row['macro_hy_spread'] = float(macro.get('macro_BAMLH0A0HYM2', 4) or 4)
    row['macro_stress'] = float(macro.get('macro_STLFSI4', 0) or 0)
    row['macro_fed_assets_chg'] = float(macro.get('macro_WALCL_chg1m', 0) or 0) / 1e12

    # News features (daily market-level, signed scale)
    news = news_data or {}
    row['news_trade_policy'] = news.get('news_trade_policy', 0)
    row['news_geopolitical'] = news.get('news_geopolitical', 0)
    row['news_regulatory'] = news.get('news_regulatory', 0)
    row['news_monetary_surprise'] = news.get('news_monetary_surprise', 0)
    row['news_energy_supply'] = news.get('news_energy_supply', 0)
    row['news_event_shock'] = news.get('news_event_shock', 0)
    row['news_sentiment_net'] = news.get('news_sentiment_net', 0)
    row['news_risk_total'] = news.get('news_risk_total', 0)
    row['news_risk_chg'] = news.get('news_risk_chg', 0)

    rsi = float(row.get('rsi', 0) or 0)
    dist_sma_200 = float(row.get('dist_sma_200', 0) or 0)
    row['inter_vix_rsi'] = macro_vix * rsi
    row['inter_vix_sma200'] = macro_vix * dist_sma_200
    row['inter_yield_sma200'] = macro_yield_curve * dist_sma_200
    row['inter_yield_rsi'] = macro_yield_curve * rsi

    div_yield = _normalize_dividend_yield(dividend_info.get('dividend_yield', 0) or 0)
    row['dividend_yield'] = div_yield
    row['is_dividend_stock'] = 1 if dividend_info.get('is_dividend') else 0
    days_to_ex_div = dividend_info.get('days_to_ex_div')
    if days_to_ex_div and days_to_ex_div > 0:
        row['days_to_ex_div_norm'] = max(0.0, 1.0 - (float(days_to_ex_div) / 30.0))
    else:
        row['days_to_ex_div_norm'] = 0.0

    payout_ratio = float(dividend_info.get('payout_ratio', 0) or 0)
    row['payout_quality'] = 1.0 if 0.3 <= payout_ratio <= 0.7 else 0.5
    row['dividend_safety_score'] = float(dividend_info.get('dividend_safety_score', 0.5) or 0.5)
    row['is_yield_trap'] = 1 if dividend_info.get('is_yield_trap') else 0
    row['dividend_growth_5y'] = float(dividend_info.get('dividend_growth_5y', 0) or 0)
    return row


def _resolve_replay_window(
    start_date: Optional[str],
    end_date: Optional[str],
    default_days: int = 365,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve replay/calibration window using DB bounds when dates are omitted."""
    from storage import get_connection

    con = get_connection(read_only=True)
    try:
        bounds = con.execute("SELECT min(date), max(date) FROM price_history").fetchone()
    finally:
        con.close()

    min_date, max_date = bounds if bounds else (None, None)
    if max_date is None:
        raise ValueError("No price_history data is available to resolve replay/calibration window.")

    min_ts = pd.to_datetime(min_date, errors='coerce').normalize()
    max_ts = pd.to_datetime(max_date, errors='coerce').normalize()
    if pd.isna(min_ts) or pd.isna(max_ts):
        raise ValueError("Failed to resolve valid date bounds from price_history.")

    end_ts = pd.to_datetime(end_date, errors='coerce') if end_date else max_ts
    if pd.isna(end_ts):
        raise ValueError("Invalid end date. Use YYYY-MM-DD.")
    end_ts = end_ts.normalize()

    if start_date:
        start_ts = pd.to_datetime(start_date, errors='coerce')
    else:
        start_ts = end_ts - pd.Timedelta(days=max(30, int(default_days)))
    if pd.isna(start_ts):
        raise ValueError("Invalid start date. Use YYYY-MM-DD.")
    start_ts = start_ts.normalize()

    if start_ts < min_ts:
        start_ts = min_ts
    if end_ts > max_ts:
        end_ts = max_ts
    if end_ts < start_ts:
        raise ValueError("Resolved replay/calibration end date is earlier than start date.")
    return start_ts, end_ts


def _load_replay_rows_for_calibration(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    horizons: List[int],
    model_version_hash: str,
    max_tickers: Optional[int] = None,
) -> pd.DataFrame:
    """Load replay predictions + realized forward closes needed for calibration fitting."""
    from storage import get_connection

    ticker_filter_sql = ""
    ticker_params: List[Any] = []
    if max_tickers and int(max_tickers) > 0:
        con = get_connection(read_only=True)
        try:
            ticker_df = con.execute(
                """
                SELECT DISTINCT upper(ticker) AS ticker
                FROM model_predictions
                WHERE asof_date BETWEEN ? AND ?
                  AND horizon IN (SELECT * FROM (SELECT UNNEST(?)))
                  AND model_version_hash = ?
                ORDER BY ticker
                """,
                [
                    start_ts.strftime('%Y-%m-%d'),
                    end_ts.strftime('%Y-%m-%d'),
                    [int(h) for h in horizons],
                    model_version_hash,
                ],
            ).df()
        finally:
            con.close()
        keep_tickers = ticker_df['ticker'].tolist()[: int(max_tickers)]
        if not keep_tickers:
            return pd.DataFrame()
        ticker_filter_sql = " AND upper(mp.ticker) IN (SELECT * FROM (SELECT UNNEST(?))) "
        ticker_params.append(keep_tickers)

    con = get_connection(read_only=True)
    try:
        df = con.execute(
            f"""
            WITH ranked_prices AS (
                SELECT
                    upper(ticker) AS ticker,
                    date,
                    close,
                    row_number() OVER (PARTITION BY upper(ticker) ORDER BY date) AS rn
                FROM price_history
            )
            SELECT
                upper(mp.ticker) AS ticker,
                mp.asof_date,
                mp.horizon,
                mp.proba_raw,
                mp.model_version_hash,
                p0.close AS asof_close,
                p1.close AS future_close
            FROM model_predictions mp
            JOIN ranked_prices p0
              ON p0.ticker = upper(mp.ticker)
             AND p0.date = mp.asof_date
            JOIN ranked_prices p1
              ON p1.ticker = p0.ticker
             AND p1.rn = p0.rn + mp.horizon
            WHERE mp.asof_date BETWEEN ? AND ?
              AND mp.horizon IN (SELECT * FROM (SELECT UNNEST(?)))
              AND mp.model_version_hash = ?
              {ticker_filter_sql}
              AND p0.close IS NOT NULL
              AND p1.close IS NOT NULL
              AND p0.close > 0
            ORDER BY mp.horizon, mp.asof_date, upper(mp.ticker)
            """,
            [
                start_ts.strftime('%Y-%m-%d'),
                end_ts.strftime('%Y-%m-%d'),
                [int(h) for h in horizons],
                model_version_hash,
                *ticker_params,
            ],
        ).df()
    finally:
        con.close()
    return df


def _latest_replay_model_hash(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    horizons: List[int],
) -> str:
    """Return most recent model_version_hash seen in replay rows for the window."""
    from storage import get_connection

    con = get_connection(read_only=True)
    try:
        row = con.execute(
            """
            SELECT model_version_hash
            FROM model_predictions
            WHERE asof_date BETWEEN ? AND ?
              AND horizon IN (SELECT * FROM (SELECT UNNEST(?)))
              AND model_version_hash IS NOT NULL
              AND model_version_hash <> ''
            GROUP BY model_version_hash
            ORDER BY max(created_at) DESC
            LIMIT 1
            """,
            [
                start_ts.strftime('%Y-%m-%d'),
                end_ts.strftime('%Y-%m-%d'),
                [int(h) for h in horizons],
            ],
        ).fetchone()
    finally:
        con.close()
    return str(row[0]) if row and row[0] else ''


def build_calibration_artifacts(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    horizons: Optional[List[int]] = None,
    max_tickers: Optional[int] = None,
    flush_rows: int = 5000,
    replay_if_missing: bool = True,
    window_days: int = 365,
) -> Dict[str, Any]:
    """
    Build per-horizon calibration artifacts from replay predictions using DB-only data.

    This path does not fetch live market data. It reuses `model_predictions` and
    `price_history`, running replay only when needed and still from DB snapshots.
    """
    from ml_engine import _calibrate_thresholds, save_model_thresholds, save_probability_calibration

    initialize_db()
    requested_horizons = sorted({int(h) for h in (horizons or HORIZONS) if int(h) > 0})
    if not requested_horizons:
        raise ValueError("No valid horizons specified for calibration artifact build.")

    start_ts, end_ts = _resolve_replay_window(start_date, end_date, default_days=window_days)
    preferred_model_hash = _compute_model_version_hash()
    model_version_hash = preferred_model_hash

    replay_df = _load_replay_rows_for_calibration(
        start_ts=start_ts,
        end_ts=end_ts,
        horizons=requested_horizons,
        model_version_hash=model_version_hash,
        max_tickers=max_tickers,
    )

    replay_refreshed = False
    if replay_df.empty and replay_if_missing:
        print(
            "Calibration build: no replay rows found for current model hash; "
            "running DB-only replay refresh first.",
            flush=True,
        )
        run_replay_scan(
            start_ts.strftime('%Y-%m-%d'),
            end_ts.strftime('%Y-%m-%d'),
            flush_rows=int(flush_rows),
            max_tickers=max_tickers,
        )
        replay_refreshed = True
        replay_df = _load_replay_rows_for_calibration(
            start_ts=start_ts,
            end_ts=end_ts,
            horizons=requested_horizons,
            model_version_hash=model_version_hash,
            max_tickers=max_tickers,
        )

    if replay_df.empty:
        fallback_hash = _latest_replay_model_hash(
            start_ts=start_ts,
            end_ts=end_ts,
            horizons=requested_horizons,
        )
        if fallback_hash and fallback_hash != model_version_hash:
            model_version_hash = fallback_hash
            replay_df = _load_replay_rows_for_calibration(
                start_ts=start_ts,
                end_ts=end_ts,
                horizons=requested_horizons,
                model_version_hash=model_version_hash,
                max_tickers=max_tickers,
            )

    if replay_df.empty:
        raise ValueError(
            "Calibration build failed: no replay prediction rows available for requested window/model hash."
        )

    model_paths = {
        5: MODEL_PATH_5D,
        10: MODEL_PATH_10D,
        30: MODEL_PATH_30D,
    }
    summary: Dict[str, Any] = {
        'start_date': start_ts.strftime('%Y-%m-%d'),
        'end_date': end_ts.strftime('%Y-%m-%d'),
        'preferred_model_version_hash': preferred_model_hash,
        'model_version_hash': model_version_hash,
        'rows_used': int(len(replay_df)),
        'replay_refreshed': bool(replay_refreshed),
        'horizons': {},
    }

    missing_horizons: List[int] = []
    for horizon in requested_horizons:
        h_df = replay_df[replay_df['horizon'] == int(horizon)].copy()
        if h_df.empty:
            missing_horizons.append(int(horizon))
            summary['horizons'][str(horizon)] = {'status': 'missing_rows'}
            continue

        raw_proba = np.clip(pd.to_numeric(h_df['proba_raw'], errors='coerce').fillna(0.5).to_numpy(), 0.0, 1.0)
        asof_close = pd.to_numeric(h_df['asof_close'], errors='coerce').to_numpy()
        future_close = pd.to_numeric(h_df['future_close'], errors='coerce').to_numpy()
        valid = np.isfinite(asof_close) & np.isfinite(future_close) & (asof_close > 0)
        if not np.any(valid):
            missing_horizons.append(int(horizon))
            summary['horizons'][str(horizon)] = {'status': 'no_valid_labels'}
            continue

        raw_valid = raw_proba[valid]
        returns = (future_close[valid] / asof_close[valid]) - 1.0
        target_return = float(HORIZON_TARGETS.get(int(horizon), 0.0) or 0.0)
        y_true = (returns >= target_return).astype(int)

        buy_thr, sell_thr, calib_metrics = _calibrate_thresholds(y_true, raw_valid)
        save_model_thresholds(int(horizon), buy_thr, sell_thr, calib_metrics)
        artifact_path = save_probability_calibration(
            horizon=int(horizon),
            raw_proba=raw_valid,
            y_true=y_true,
            model_path=model_paths.get(int(horizon), ''),
            model_status={
                'mode': 'replay_calibration_build',
                'start_date': start_ts.strftime('%Y-%m-%d'),
                'end_date': end_ts.strftime('%Y-%m-%d'),
                'target_return': target_return,
                'rows': int(len(raw_valid)),
                'model_version_hash': model_version_hash,
            },
        )
        summary['horizons'][str(horizon)] = {
            'status': 'ok',
            'rows': int(len(raw_valid)),
            'target_return': target_return,
            'buy_threshold': float(buy_thr),
            'sell_threshold': float(sell_thr),
            'artifact_path': artifact_path,
        }

    if missing_horizons:
        raise ValueError(
            f"Calibration build incomplete: missing usable rows for horizons {missing_horizons} "
            f"in window {start_ts.strftime('%Y-%m-%d')}..{end_ts.strftime('%Y-%m-%d')}."
        )

    print(f"Calibration artifacts built: {summary}", flush=True)
    return summary


def run_replay_scan(start_date: str, end_date: str, flush_rows: int = 5000, max_tickers: int = None) -> dict:
    """
    Replay scanner-style inference as-of each historical date and save model predictions.
    """
    initialize_db()

    start_ts = pd.to_datetime(start_date, errors='coerce')
    end_ts = pd.to_datetime(end_date, errors='coerce')
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError("Invalid replay date(s). Use YYYY-MM-DD format.")
    start_ts = start_ts.normalize()
    end_ts = end_ts.normalize()
    if end_ts < start_ts:
        raise ValueError("Replay end date must be >= start date.")

    from storage import get_connection
    from ml_engine import extract_ml_features, get_multi_horizon_confidence, load_qual_features
    from macro_loader import fetch_all_macro_data, get_macro_features_asof
    from news_loader import load_all_news_cache, get_news_features_asof

    con = get_connection(read_only=True)
    replay_dates_df = con.execute(
        """
        SELECT DISTINCT date
        FROM price_history
        WHERE date BETWEEN ? AND ?
        ORDER BY date
        """,
        [start_ts.strftime('%Y-%m-%d'), end_ts.strftime('%Y-%m-%d')],
    ).df()
    if replay_dates_df.empty:
        con.close()
        print("Replay scan: no trading dates found in range.")
        return {
            'start_date': start_ts.strftime('%Y-%m-%d'),
            'end_date': end_ts.strftime('%Y-%m-%d'),
            'replay_dates': 0,
            'tickers': 0,
            'prediction_rows': 0,
            'model_version_hash': '',
        }

    tickers_df = con.execute(
        """
        SELECT DISTINCT upper(ticker) AS ticker
        FROM price_history
        WHERE date <= ?
        ORDER BY ticker
        """,
        [end_ts.strftime('%Y-%m-%d')],
    ).df()
    con.close()

    tickers = [t for t in tickers_df['ticker'].tolist() if t and t not in UNIVERSE_DENYSET]
    if max_tickers and max_tickers > 0:
        tickers = tickers[:int(max_tickers)]
    if not tickers:
        print("Replay scan: no eligible tickers available in DB.")
        return {
            'start_date': start_ts.strftime('%Y-%m-%d'),
            'end_date': end_ts.strftime('%Y-%m-%d'),
            'replay_dates': 0,
            'tickers': 0,
            'prediction_rows': 0,
            'model_version_hash': '',
        }

    con = get_connection(read_only=True)
    price_df = con.execute(
        """
        SELECT upper(ticker) AS ticker, date, open, high, low, close, volume
        FROM price_history
        WHERE ticker IN (SELECT * FROM (SELECT UNNEST(?)))
          AND date <= ?
        ORDER BY ticker, date
        """,
        [tickers, end_ts.strftime('%Y-%m-%d')],
    ).df()
    con.close()

    if price_df.empty:
        print("Replay scan: no price data found.")
        return {
            'start_date': start_ts.strftime('%Y-%m-%d'),
            'end_date': end_ts.strftime('%Y-%m-%d'),
            'replay_dates': 0,
            'tickers': 0,
            'prediction_rows': 0,
            'model_version_hash': '',
        }

    price_df['date'] = pd.to_datetime(price_df['date'], errors='coerce').dt.normalize()
    price_df = price_df.dropna(subset=['date', 'ticker']).copy()

    try:
        qual_cache = load_qual_features() or {}
    except Exception as e:
        print(f"Replay scan: qual feature load failed ({e}); using defaults.")
        qual_cache = {}

    thresholds_5d = get_model_thresholds(5)
    thresholds_10d = get_model_thresholds(10)
    thresholds_30d = get_model_thresholds(30)
    model_version_hash = _compute_model_version_hash()
    dividend_history_cache, fundamentals_cache = _load_replay_dividend_context(tickers, end_ts)
    try:
        macro_series_cache = fetch_all_macro_data()
    except Exception as e:
        print(f"Replay scan: macro series load failed ({e}); using empty defaults.")
        macro_series_cache = {}
    macro_asof_cache: Dict[pd.Timestamp, Dict[str, float]] = {}

    try:
        news_cache = load_all_news_cache()
        if news_cache:
            print(f"Replay scan: loaded news cache for {len(news_cache)} dates.", flush=True)
    except Exception as e:
        print(f"Replay scan: news cache load failed ({e}); using defaults.")
        news_cache = {}

    ticker_cache = {}
    for ticker, group in price_df.groupby('ticker', sort=True):
        if len(group) < 200:
            continue
        try:
            signal_df = generate_signals(group.reset_index(drop=True), ticker=ticker)
        except Exception as e:
            print(f"Replay scan: generate_signals failed for {ticker}: {e}")
            continue

        if signal_df.empty:
            continue

        signal_df = signal_df.copy()
        signal_df['date'] = pd.to_datetime(signal_df['date'], errors='coerce').dt.normalize()
        signal_df = signal_df.dropna(subset=['date']).reset_index(drop=True)
        if signal_df.empty:
            continue

        qual_data = qual_cache.get(ticker.upper()) if qual_cache else None
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
            feature_df = extract_ml_features(
                signal_df,
                dividend_info={},
                macro_data={},
                qual_data=qual_data,
                news_data={},
                horizon=10,
                for_inference=True,
            )
        if feature_df.empty:
            continue

        feature_dates = pd.to_datetime(signal_df.loc[feature_df.index, 'date'], errors='coerce').dt.normalize()
        feature_df = feature_df.reset_index(drop=True)
        feature_dates = feature_dates.reset_index(drop=True)
        if feature_dates.empty:
            continue

        ticker_cache[ticker] = {
            'signal_df': signal_df,
            'signal_dates': signal_df['date'].to_numpy(),
            'feature_df': feature_df,
            'feature_dates': feature_dates.to_numpy(),
        }

    replay_dates = pd.to_datetime(replay_dates_df['date'], errors='coerce').dropna().dt.normalize().tolist()
    replay_rows = []
    total_saved = 0

    print(
        f"Replay scan start: dates={len(replay_dates)}, tickers={len(ticker_cache)}, "
        f"range={start_ts.strftime('%Y-%m-%d')}..{end_ts.strftime('%Y-%m-%d')}"
    )

    for idx_day, asof_date in enumerate(replay_dates, 1):
        if idx_day == 1 or idx_day % 10 == 0 or idx_day == len(replay_dates):
            print(f"Replay progress: {idx_day}/{len(replay_dates)} dates ({asof_date.strftime('%Y-%m-%d')})")

        asof_np = asof_date.to_datetime64()
        asof_macro = macro_asof_cache.get(asof_date)
        if asof_macro is None:
            try:
                asof_macro = get_macro_features_asof(asof_date, data=macro_series_cache)
            except Exception:
                asof_macro = {}
            macro_asof_cache[asof_date] = asof_macro

        for ticker, data in ticker_cache.items():
            signal_dates = data['signal_dates']
            sig_idx = int(np.searchsorted(signal_dates, asof_np, side='right'))
            if sig_idx < 200:
                continue

            signal_slice = data['signal_df'].iloc[:sig_idx]
            if signal_slice.empty:
                continue

            last_row = signal_slice.iloc[-1]
            last_dt = pd.to_datetime(last_row['date']).normalize()
            age_days = int((asof_date - last_dt).days)
            if age_days > MAX_PRICE_DATA_STALENESS_DAYS and not ALLOW_STALE_DATA_FOR_SIGNALS:
                continue

            feature_dates = data['feature_dates']
            feat_idx = int(np.searchsorted(feature_dates, asof_np, side='right'))
            if feat_idx < 1:
                continue

            dividend_info = _resolve_replay_dividend_info(
                ticker,
                asof_date,
                asof_np,
                dividend_history_cache,
                fundamentals_cache,
            )
            asof_news = get_news_features_asof(asof_date, news_cache=news_cache)
            feature_row = _apply_replay_context_to_feature_row(
                data['feature_df'].iloc[feat_idx - 1],
                asof_macro,
                dividend_info,
                news_data=asof_news,
            )
            try:
                horizon_confs = get_multi_horizon_confidence(feature_row, ticker=ticker)
            except Exception:
                horizon_confs = {'conf_5d': 0.5, 'conf_10d': 0.5, 'conf_30d': 0.5}

            conf_5d = float(horizon_confs.get('conf_5d', 0.5))
            conf_10d = float(horizon_confs.get('conf_10d', 0.5))
            conf_30d = float(horizon_confs.get('conf_30d', 0.5))
            raw_5d = float(horizon_confs.get('raw_conf_5d', conf_5d))
            raw_10d = float(horizon_confs.get('raw_conf_10d', conf_10d))
            raw_30d = float(horizon_confs.get('raw_conf_30d', conf_30d))

            buy_thr_5d = float(thresholds_5d.get('buy', 0.60))
            sell_thr_5d = float(thresholds_5d.get('sell', 0.40))
            buy_thr_10d = float(thresholds_10d.get('buy', 0.60))
            sell_thr_10d = float(thresholds_10d.get('sell', 0.40))
            buy_thr_30d = float(thresholds_30d.get('buy', 0.60))
            sell_thr_30d = float(thresholds_30d.get('sell', 0.40))

            strong_buy = (
                conf_5d >= max(buy_thr_5d, CONFLUENCE_THRESHOLD)
                and conf_10d >= max(buy_thr_10d, CONFLUENCE_THRESHOLD)
                and conf_30d >= max(buy_thr_30d, CONFLUENCE_THRESHOLD)
            )
            tactical_buy = (
                conf_5d >= max(buy_thr_5d, TACTICAL_THRESHOLD)
                and conf_10d >= buy_thr_10d
            )
            trend_buy = (
                conf_30d >= max(buy_thr_30d, TREND_THRESHOLD)
                and conf_10d >= buy_thr_10d
            )
            confirmed_sell = (
                conf_10d <= sell_thr_10d
                and (conf_5d <= sell_thr_5d or conf_30d <= sell_thr_30d)
            )

            rec_status = "HOLD"
            if strong_buy or tactical_buy or trend_buy:
                rec_status = "BUY"
            elif confirmed_sell:
                rec_status = "SELL"

            price_at_rec = float(last_row.get('close', 0.0) or 0.0)
            atr_ratio_val = float(last_row.get('atr_ratio', 0.0) or 0.0)
            tradable, tradability_reason, _, _ = _evaluate_tradability(
                ticker=ticker,
                ticker_df=signal_slice,
                price=price_at_rec,
                atr_ratio=atr_ratio_val,
            )

            filters_triggered = []
            if age_days > 0:
                filters_triggered.append(f"stale_data:{age_days}d")
            if rec_status == "BUY" and not tradable:
                rec_status = "HOLD"
                filters_triggered.append(f"tradability:{tradability_reason}")

            snapshot_hash = _compute_data_snapshot_hash(ticker, asof_date, signal_slice)
            filter_str = ";".join(filters_triggered)

            cal_map = {5: conf_5d, 10: conf_10d, 30: conf_30d}
            raw_map = {5: raw_5d, 10: raw_10d, 30: raw_30d}
            for horizon, proba_cal in cal_map.items():
                proba_raw = raw_map.get(int(horizon), proba_cal)
                replay_rows.append({
                    'asof_date': asof_date.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'horizon': int(horizon),
                    'proba_raw': float(proba_raw),
                    'proba_cal': float(proba_cal),
                    'score': float(proba_cal),
                    'signal_type': rec_status,
                    'tradable': bool(tradable),
                    'tradability_reason': tradability_reason,
                    'filters_triggered': filter_str,
                    'model_version_hash': model_version_hash,
                    'data_snapshot_hash': snapshot_hash,
                })

            if len(replay_rows) >= int(flush_rows):
                save_model_predictions(replay_rows)
                total_saved += len(replay_rows)
                replay_rows.clear()

    if replay_rows:
        save_model_predictions(replay_rows)
        total_saved += len(replay_rows)
        replay_rows.clear()

    summary = {
        'start_date': start_ts.strftime('%Y-%m-%d'),
        'end_date': end_ts.strftime('%Y-%m-%d'),
        'replay_dates': len(replay_dates),
        'tickers': len(ticker_cache),
        'prediction_rows': total_saved,
        'model_version_hash': model_version_hash,
    }
    print(f"Replay complete: {summary}")
    return summary
	
def run_replay_reproducibility_check(
    start_date: str,
    end_date: str,
    replay_runs: int = 2,
    max_tickers: int | None = None,
    flush_rows: int = 5000,
    report_path: str | None = None,
) -> Dict[str, Any]:
    """
    Run replay repeatedly for the same range and compare row hashes for reproducibility.
    """
    if replay_runs < 2:
        raise ValueError("replay_runs must be at least 2.")

    start_ts = pd.to_datetime(start_date, errors='coerce')
    end_ts = pd.to_datetime(end_date, errors='coerce')
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError("Invalid replay date(s). Use YYYY-MM-DD format.")
    start_ts = start_ts.normalize()
    end_ts = end_ts.normalize()
    if end_ts < start_ts:
        raise ValueError("Replay end date must be >= start date.")

    run_summaries: List[Dict[str, Any]] = []
    snapshots: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    model_version_hash = ''
    fatal_error = None

    for run_idx in range(1, int(replay_runs) + 1):
        print(f"Replay validation run {run_idx}/{replay_runs}: executing replay scan...")
        run_summary = run_replay_scan(
            start_date=start_ts.strftime('%Y-%m-%d'),
            end_date=end_ts.strftime('%Y-%m-%d'),
            flush_rows=flush_rows,
            max_tickers=max_tickers,
        )
        run_summaries.append(run_summary)

        current_hash = _coerce_string(run_summary.get('model_version_hash'), default='').strip()
        if run_idx == 1:
            model_version_hash = current_hash
        elif current_hash != model_version_hash:
            fatal_error = (
                f"Model hash changed between runs: run1={model_version_hash}, "
                f"run{run_idx}={current_hash}"
            )
            break

        snapshot = _snapshot_replay_predictions(start_ts, end_ts, model_version_hash)
        snapshots.append(snapshot)

    if fatal_error:
        status = 'FAIL'
    elif len(snapshots) < 2:
        status = 'FAIL'
        if not fatal_error:
            fatal_error = (
                f"Insufficient replay snapshots collected: expected {replay_runs}, "
                f"collected {len(snapshots)}"
            )
    else:
        for snapshot_idx in range(1, len(snapshots)):
            comparison = _compare_replay_snapshots(snapshots[0], snapshots[snapshot_idx])
            comparison['comparison_run'] = snapshot_idx + 1
            comparisons.append(comparison)
        status = 'PASS' if all(c.get('passes', False) for c in comparisons) else 'FAIL'

    last_comparison = comparisons[0] if comparisons else {}
    result: Dict[str, Any] = {
        'status': status,
        'start_date': start_ts.strftime('%Y-%m-%d'),
        'end_date': end_ts.strftime('%Y-%m-%d'),
        'replay_runs': replay_runs,
        'model_version_hash': model_version_hash,
        'run_summaries': run_summaries,
        'snapshots': snapshots,
        'comparisons': comparisons,
        'rows_compared': last_comparison.get('rows_compared', 0),
        'mismatch_count': last_comparison.get('mismatch_count', 0),
        'sample_keys': last_comparison.get('sample_keys', []),
        'fatal_error': fatal_error,
    }

    try:
        result['report_path'] = _write_replay_validation_report(result, report_path=report_path)
        print(f"Replay reproducibility report: {result['report_path']}")
    except Exception as report_error:
        print(f"Replay report write failed: {report_error}")
        result['report_path'] = None

    print(
        f"Replay determinism check: {result['status']} | "
        f"rows_compared={result['rows_compared']} | "
        f"mismatches={result['mismatch_count']}"
    )
    if fatal_error:
        print(f"Replay determinism check failed: {fatal_error}")

    return result


if __name__ == "__main__":
    run_full_scan()
