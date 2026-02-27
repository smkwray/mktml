"""
News Loader
============
Fetches daily market-level news assessment via Gemini CLI (grounded search).
Produces structured signed risk scores for trade policy, geopolitical,
regulatory, monetary, energy/commodity, event shock, and net sentiment signals
-- things not captured by FRED/VIX/yield curve.

Architecture:
  - Two calls per day: morning (pre-market) + evening (post-close)
  - Evening call overwrites morning cache (more complete assessment)
  - Cached per-date for replay/backtesting
  - Graceful failure: returns neutral defaults if Gemini unavailable
  - Computed features (risk_total, risk_chg) derived from raw scores
"""
import os
import sys
import json
import re
import subprocess
from datetime import datetime, date, timedelta
from typing import Dict, Optional

# Project root resolution (same pattern as macro_loader)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_SCRIPT_DIR) == 'src':
    PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
else:
    PROJECT_ROOT = _SCRIPT_DIR

sys.path.insert(0, PROJECT_ROOT)
from config import (
    NEWS_ENABLED,
    NEWS_CACHE_DIR,
    GEMINI_NEWS_MODEL,
    GEMINI_NEWS_TIMEOUT,
)
try:
    from config import GEMINI_NEWS_FALLBACK_MODEL
except ImportError:
    GEMINI_NEWS_FALLBACK_MODEL = None

# ── Feature schema ──────────────────────────────────────────────
# Raw scores from Gemini (cached in JSON)
_RAW_SCORE_KEYS = [
    'news_trade_policy',      # -3 to +3  (signed: neg=risk, pos=opportunity)
    'news_geopolitical',      # -3 to +3
    'news_regulatory',        # -3 to +3
    'news_monetary_surprise', # -3 to +3  (unexpected central bank moves)
    'news_energy_supply',     # -3 to +3  (commodity/OPEC shocks)
    'news_event_shock',       #  0 to  3  (always non-negative)
    'news_sentiment_net',     # -5 to +5  (overall market tilt)
]

# Computed features (derived in Python, not from Gemini)
_COMPUTED_KEYS = [
    'news_risk_chg',          # today's risk_total minus yesterday's
    'news_risk_total',        # sum of absolute raw scores (composite intensity)
]

# Neutral defaults when news data is unavailable
NEWS_DEFAULTS = {k: 0 for k in _RAW_SCORE_KEYS + _COMPUTED_KEYS}

# Validation ranges for raw scores: key -> (min, max)
_SCORE_RANGES = {
    'news_trade_policy':      (-3, 3),
    'news_geopolitical':      (-3, 3),
    'news_regulatory':        (-3, 3),
    'news_monetary_surprise': (-3, 3),
    'news_energy_supply':     (-3, 3),
    'news_event_shock':       (0, 3),
    'news_sentiment_net':     (-5, 5),
}


# ── Cache helpers ───────────────────────────────────────────────

def _cache_path(target_date: date) -> str:
    """Return cache file path for a given date."""
    os.makedirs(NEWS_CACHE_DIR, exist_ok=True)
    return os.path.join(NEWS_CACHE_DIR, f'{target_date.isoformat()}.json')


def _has_cache(target_date: date) -> bool:
    """Check if a cache file exists for the given date."""
    return os.path.exists(_cache_path(target_date))


def _load_cache(target_date: date) -> Optional[dict]:
    """Load cached news data for a date. Returns None on any failure."""
    path = _cache_path(target_date)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        # Validate at least the core fields exist
        for key in _RAW_SCORE_KEYS:
            if key not in data:
                return None
        return data
    except Exception:
        return None


def _save_cache(target_date: date, data: dict):
    """Save news data to cache file."""
    path = _cache_path(target_date)
    data['_meta'] = {
        'cached_at': datetime.now().isoformat(),
        'model': GEMINI_NEWS_MODEL,
    }
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[news] Warning: Could not save cache: {e}")


# ── Computed features ───────────────────────────────────────────

def _compute_risk_total(data: dict) -> float:
    """Sum of absolute raw scores — overall news intensity regardless of direction."""
    return sum(abs(data.get(k, 0)) for k in _RAW_SCORE_KEYS)


def _compute_risk_chg(target_date: date, today_total: float,
                      news_cache: dict = None) -> float:
    """
    Change in risk_total from previous day's cache.
    Returns 0 if no previous day data (expected for first day of caching).
    """
    yesterday = target_date - timedelta(days=1)

    # Try pre-loaded replay cache first
    prev_data = None
    if news_cache:
        prev_data = news_cache.get(yesterday.isoformat())

    # Fall back to file cache
    if prev_data is None:
        prev_data = _load_cache(yesterday)

    if prev_data is None:
        return 0.0

    prev_total = sum(abs(prev_data.get(k, 0)) for k in _RAW_SCORE_KEYS)
    return today_total - prev_total


def _enrich_with_computed(data: dict, target_date: date,
                          news_cache: dict = None) -> dict:
    """Add computed features (risk_total, risk_chg) to a raw data dict."""
    out = dict(data)
    risk_total = _compute_risk_total(out)
    out['news_risk_total'] = risk_total
    out['news_risk_chg'] = _compute_risk_chg(target_date, risk_total, news_cache)
    return out


# ── Gemini prompt ───────────────────────────────────────────────

def _generate_prompt(target_date: date) -> str:
    """Generate the Gemini prompt for daily market news assessment."""
    return f'''Search online for today's major market-moving news and events as of {target_date.isoformat()}.

Focus ONLY on events NOT already captured by standard market indicators (VIX, yield curve, credit spreads, financial stress index, Fed balance sheet, consumer sentiment, jobless claims).

Assess each category on a SIGNED scale where NEGATIVE = bearish/risk, POSITIVE = bullish/opportunity:

1. Trade policy (-3 to +3): tariffs, sanctions, trade deals, embargoes. Negative = new restrictions/tariffs. Positive = trade deal progress/tariff removal.
2. Geopolitical (-3 to +3): wars, conflicts, diplomatic crises, elections. Negative = escalation/crisis. Positive = peace deal/resolution/de-escalation.
3. Regulatory (-3 to +3): legislation, antitrust, sector regulations. Negative = restrictive new regulation. Positive = deregulation/business-friendly policy.
4. Monetary policy surprise (-3 to +3): unexpected central bank actions NOT already in Fed funds futures or yield curve. Negative = hawkish surprise. Positive = dovish surprise. 0 if nothing unexpected.
5. Energy/commodity supply (-3 to +3): OPEC decisions, pipeline disruptions, commodity supply shocks. Negative = supply disruption/price spike. Positive = supply expansion/price relief.
6. Event shock (0 to 3): pandemics, natural disasters, flash crashes, infrastructure failures, cyberattacks. 0=none, 1=minor, 2=significant, 3=severe.
7. Net market sentiment (-5 to +5): your overall assessment of how today's news environment tilts equity markets beyond what VIX/spreads capture. Negative = risk-off, Positive = risk-on.

Scoring guide for signed categories:
-3 = severe negative (e.g., major new tariffs enacted, active war escalation)
-2 = significant negative (e.g., sanctions announced, conflict intensifying)
-1 = mild negative (e.g., trade talks stalled, tensions rising)
 0 = neutral or nothing notable
+1 = mild positive (e.g., talks resumed, tensions easing)
+2 = significant positive (e.g., deal framework announced, ceasefire)
+3 = strong positive (e.g., major trade deal signed, peace agreement)

Return ONLY valid JSON with no markdown, no code blocks, no commentary:
{{
  "date": "{target_date.isoformat()}",
  "news_trade_policy": 0,
  "news_geopolitical": 0,
  "news_regulatory": 0,
  "news_monetary_surprise": 0,
  "news_energy_supply": 0,
  "news_event_shock": 0,
  "news_sentiment_net": 0,
  "sectors_affected": ["sector1", "sector2"],
  "justification": {{
    "trade_policy": "brief reason or 'none'",
    "geopolitical": "brief reason or 'none'",
    "regulatory": "brief reason or 'none'",
    "monetary_surprise": "brief reason or 'none'",
    "energy_supply": "brief reason or 'none'",
    "event_shock": "brief reason or 'none'",
    "sentiment_net": "brief reason"
  }},
  "sources": [
    {{"title": "Source name", "url": "https://..."}}
  ]
}}

For sectors_affected, list the 1-3 GICS sectors MOST impacted by today's news (from: Energy, Materials, Industrials, Consumer Discretionary, Consumer Staples, Health Care, Financials, Information Technology, Communication Services, Utilities, Real Estate). Empty list if no sector-specific impact.

You MUST search online for current information. Do not rely on training data alone.'''


# ── Gemini CLI execution ───────────────────────────────────────

def _find_gemini_binary() -> Optional[str]:
    """Find gemini CLI binary in standard locations."""
    paths = [
        "gemini",
        "/opt/homebrew/bin/gemini",
        "/usr/local/bin/gemini",
        os.path.expanduser("~/.local/bin/gemini"),
    ]
    for path in paths:
        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _run_gemini_once(prompt: str, model: Optional[str] = None) -> Optional[str]:
    """Run Gemini CLI with a specific model and return raw output. Returns None on failure."""
    gemini_bin = _find_gemini_binary()
    if not gemini_bin:
        print("[news] ERROR: gemini-cli not found in standard paths.")
        return None

    cmd = [gemini_bin]
    if model:
        cmd.extend(["-m", model])
    cmd.extend(["-p", prompt])

    label = model or "default"
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=GEMINI_NEWS_TIMEOUT,
        )
        combined_lower = (result.stdout + result.stderr).lower()

        # Check for rate limit / quota / model-not-found errors
        fail_keywords = [
            'quota exceeded', 'resource exhausted', 'too many requests', '429',
            'not found', 'model not found',
        ]
        if result.returncode != 0 and any(k in combined_lower for k in fail_keywords):
            print(f"[news] API error on {label} (exit={result.returncode}).")
            return None

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

        if result.stderr:
            print(f"[news] Gemini stderr ({label}): {result.stderr.strip()[:200]}")
        return None

    except subprocess.TimeoutExpired:
        print(f"[news] Timeout after {GEMINI_NEWS_TIMEOUT}s ({label}).")
        return None
    except Exception as e:
        print(f"[news] Error running gemini ({label}): {e}")
        return None


def _run_gemini(prompt: str) -> Optional[str]:
    """Run Gemini CLI with automatic fallback. Tries primary model first, then fallback."""
    result = _run_gemini_once(prompt, model=GEMINI_NEWS_MODEL)
    if result:
        return result

    if GEMINI_NEWS_FALLBACK_MODEL and GEMINI_NEWS_FALLBACK_MODEL != GEMINI_NEWS_MODEL:
        print(f"[news] Primary model ({GEMINI_NEWS_MODEL}) failed. Trying fallback ({GEMINI_NEWS_FALLBACK_MODEL})...")
        return _run_gemini_once(prompt, model=GEMINI_NEWS_FALLBACK_MODEL)

    return None


# ── Response parsing ────────────────────────────────────────────

def _parse_response(raw: str) -> Optional[dict]:
    """Parse and validate Gemini JSON response."""
    if not raw:
        return None

    # Extract JSON from response (strip any markdown fencing)
    cleaned = raw.strip()
    if cleaned.startswith('```'):
        lines = cleaned.split('\n')
        lines = [l for l in lines if not l.strip().startswith('```')]
        cleaned = '\n'.join(lines)

    # Find JSON object boundaries
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start < 0 or end <= start:
        print("[news] Could not find JSON object in response.")
        return None

    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as e:
        print(f"[news] JSON parse error: {e}")
        return None

    # Validate and clamp scores to expected ranges
    for key, (lo, hi) in _SCORE_RANGES.items():
        val = data.get(key)
        if val is None or not isinstance(val, (int, float)):
            # Try alternate key names (without news_ prefix)
            alt_key = key.replace('news_', '')
            val = data.get(alt_key)
            if val is not None:
                data[key] = val

        try:
            data[key] = max(lo, min(int(data.get(key, 0)), hi))
        except (TypeError, ValueError):
            data[key] = 0

    return data


# ── Public API ──────────────────────────────────────────────────

def fetch_daily_news(target_date: date = None, force: bool = False) -> dict:
    """
    Fetch daily market news assessment via Gemini CLI.

    Args:
        target_date: Date to assess (defaults to today)
        force: If True, overwrite existing cache (used for evening update)

    Returns:
        Dict with all news features (raw + computed)
    """
    if not NEWS_ENABLED:
        print("[news] News loader disabled in config.")
        return dict(NEWS_DEFAULTS)

    if target_date is None:
        target_date = date.today()

    # Check cache (skip if forcing refresh for evening update)
    if not force and _has_cache(target_date):
        cached = _load_cache(target_date)
        if cached:
            print(f"[news] Using cached assessment for {target_date}.")
            return _enrich_with_computed(
                {k: cached.get(k, 0) for k in _RAW_SCORE_KEYS},
                target_date,
            )

    print(f"[news] Fetching market news assessment for {target_date} via Gemini CLI...")
    prompt = _generate_prompt(target_date)
    raw = _run_gemini(prompt)

    if raw is None:
        print("[news] Gemini call failed. Using neutral defaults.")
        return dict(NEWS_DEFAULTS)

    parsed = _parse_response(raw)
    if parsed is None:
        print("[news] Could not parse response. Using neutral defaults.")
        return dict(NEWS_DEFAULTS)

    # Cache the result
    _save_cache(target_date, parsed)

    result = _enrich_with_computed(
        {k: parsed.get(k, 0) for k in _RAW_SCORE_KEYS},
        target_date,
    )
    print(f"[news] Assessment: trade={result['news_trade_policy']} "
          f"geo={result['news_geopolitical']} reg={result['news_regulatory']} "
          f"monetary={result['news_monetary_surprise']} energy={result['news_energy_supply']} "
          f"shock={result['news_event_shock']} sentiment={result['news_sentiment_net']} "
          f"risk_total={result['news_risk_total']}")
    return result


def get_news_features(target_date: date = None) -> Dict[str, float]:
    """
    Get news features for ML model. Loads from cache if available,
    otherwise returns neutral defaults.

    This is the primary interface for ml_engine and scanner.

    Args:
        target_date: Date to get features for (defaults to today)

    Returns:
        Dict of feature_name -> value (raw + computed keys)
    """
    if not NEWS_ENABLED:
        return dict(NEWS_DEFAULTS)

    if target_date is None:
        target_date = date.today()

    cached = _load_cache(target_date)
    if cached:
        return _enrich_with_computed(
            {k: cached.get(k, 0) for k in _RAW_SCORE_KEYS},
            target_date,
        )

    # No cache for this date -- return defaults (don't auto-fetch during inference)
    return dict(NEWS_DEFAULTS)


def get_news_features_asof(asof_date, news_cache: dict = None) -> Dict[str, float]:
    """
    Get news features for a historical date (replay mode).

    Checks the date-keyed cache. Returns defaults if no cache exists
    for that date (expected for dates before news caching started).

    Args:
        asof_date: Date to get features for (str or date-like)
        news_cache: Optional pre-loaded cache dict {date_str: data}

    Returns:
        Dict of feature_name -> value (raw + computed)
    """
    if not NEWS_ENABLED:
        return dict(NEWS_DEFAULTS)

    try:
        if isinstance(asof_date, str):
            target = date.fromisoformat(asof_date)
        elif hasattr(asof_date, 'date'):
            target = asof_date.date()
        else:
            target = asof_date
    except Exception:
        return dict(NEWS_DEFAULTS)

    # Check pre-loaded cache first (for replay efficiency)
    if news_cache:
        data = news_cache.get(target.isoformat())
        if data:
            return _enrich_with_computed(
                {k: data.get(k, 0) for k in _RAW_SCORE_KEYS},
                target,
                news_cache=news_cache,
            )

    # Fall back to file cache
    cached = _load_cache(target)
    if cached:
        return _enrich_with_computed(
            {k: cached.get(k, 0) for k in _RAW_SCORE_KEYS},
            target,
        )

    return dict(NEWS_DEFAULTS)


def load_all_news_cache() -> dict:
    """
    Load all cached news data into memory for replay scans.

    Returns:
        Dict mapping date_str -> news data dict
    """
    cache = {}
    if not os.path.isdir(NEWS_CACHE_DIR):
        return cache
    for fname in os.listdir(NEWS_CACHE_DIR):
        if not fname.endswith('.json'):
            continue
        date_str = fname.replace('.json', '')
        try:
            with open(os.path.join(NEWS_CACHE_DIR, fname), 'r') as f:
                data = json.load(f)
            cache[date_str] = data
        except Exception:
            continue
    return cache


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fetch daily market news assessment')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--force', action='store_true', help='Overwrite existing cache')
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else None
    result = fetch_daily_news(target_date=target, force=args.force)
    print(f"\nResult: {json.dumps(result, indent=2)}")
