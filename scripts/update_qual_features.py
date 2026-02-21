#!/usr/bin/env python3
"""
Update Qualitative Features via Gemini CLI
==========================================
One ticker at a time with explicit online search and source validation.

Usage:
    python scripts/update_qual_features.py              # Process next chunk
    python scripts/update_qual_features.py --ticker AAPL  # Single ticker
    python scripts/update_qual_features.py --count 50    # Override chunk size
"""

import os
import sys
import json
import subprocess
import argparse
import time
from datetime import datetime, timedelta
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from config import PORTFOLIO_HOLDINGS, WATCHLIST

QUAL_FEATURES_FILE = os.path.join(PROJECT_ROOT, 'data', 'qualitative_features.json')
QUAL_SETTINGS_FILE = os.path.join(PROJECT_ROOT, 'data', 'qual_settings.json')
LOG_FILE = os.path.join(PROJECT_ROOT, 'qual.log')
DEFAULT_CHUNK_SIZE = 20
# Freshness targets by priority tier
STALE_DAYS_PORTFOLIO = 7
STALE_DAYS_WATCHLIST = 14
STALE_DAYS_UNIVERSE = 45
DELAY_BETWEEN_CALLS = 2  # seconds between gemini calls
PRIMARY_MODEL = "gemini-3-flash-preview"  # Use flash as primary (faster, less rate-limited)
FALLBACK_MODEL = None  # None means use default gemini-cli (no -m flag)
AUTO_SWITCH_PROMPTS = 30  # Auto-switch to fallback after this many prompts

# Valid taxonomies (must match exactly)
VALID_SECTORS = [
    "Energy", "Materials", "Industrials", "Consumer Discretionary",
    "Consumer Staples", "Healthcare", "Financials", "Technology",
    "Communication Services", "Utilities", "Real Estate", "Diversified", "Unknown"
]

VALID_INDUSTRIES = [
    "Oil and Gas", "Energy Equipment", "Chemicals", "Metals and Mining",
    "Construction Materials", "Aerospace and Defense", "Industrial Machinery",
    "Transportation", "Retail", "Automotive", "Hotels and Leisure",
    "Media and Entertainment", "Food and Beverage", "Household Products",
    "Pharmaceuticals", "Biotech", "Healthcare Equipment", "Healthcare Services",
    "Banks", "Insurance", "Asset Management", "Software",
    "Hardware and Semiconductors", "IT Services", "Telecom", "Utilities",
    "REITs", "Real Estate Services", "Diversified ETF", "Bond ETF",
    "Sector ETF", "International ETF", "Unknown"
]

VALID_MATURITY = ["startup", "growth", "mature", "declining"]
VALID_CYCLICAL = ["defensive", "mixed", "cyclical"]
VALID_MOAT = ["none", "narrow", "wide"]
VALID_DEBT = ["low", "medium", "high"]
VALID_VALUATION_RISK = ["low", "medium", "high"]
VALID_RATES_SENSITIVITY = ["low", "medium", "high"]
VALID_COMMODITY_SENSITIVITY = ["low", "medium", "high"]
VALID_FX_SENSITIVITY = ["low", "medium", "high"]


def log_entry(msg: str, total_tickers: int = None):
    """Append a log entry to qual.log. Opens/writes/closes to avoid file locking."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    suffix = f" | total={total_tickers}" if total_tickers is not None else ""
    line = f"[{timestamp}] {msg}{suffix}\n"
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line)
            f.flush()
    except Exception as e:
        print(f"[update_qual] WARNING: Could not write to log: {e}")


def get_all_tickers() -> list:
    """Get all tickers from the scanner universe."""
    try:
        from src.universe import get_master_universe
        return get_master_universe()
    except ImportError as e:
        print(f"[update_qual] WARNING: Could not import src.universe: {e}")
    
    # Fallback: combine known lists
    from config import PORTFOLIO_HOLDINGS, WATCHLIST
    tickers = list(set(PORTFOLIO_HOLDINGS + WATCHLIST))
    print(f"[update_qual] WARNING: Using fallback ticker list ({len(tickers)} tickers)")
    return tickers



def load_settings() -> dict:
    """Load settings from settings file."""
    defaults = {
        'enabled': True,
        'chunk_size': 12,
        'interval_hours': 4,
        'stale_days_portfolio': STALE_DAYS_PORTFOLIO,
        'stale_days_watchlist': STALE_DAYS_WATCHLIST,
        'stale_days_universe': STALE_DAYS_UNIVERSE,
    }
    try:
        with open(QUAL_SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
        return {**defaults, **settings}
    except FileNotFoundError:
        return defaults


def save_settings(settings: dict):
    """Save settings to file."""
    with open(QUAL_SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)


def load_existing_data() -> dict:
    """Load existing qualitative features."""
    try:
        with open(QUAL_FEATURES_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "_meta": {
                "description": "Qualitative/industry features via gemini-cli",
                "schema_version": "4.0",
                "prompt_version": "v4-strict-taxonomy"
            },
            "tickers": {}
        }


def save_data(data: dict):
    """Save qualitative features to file."""
    data["_meta"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data["_meta"]["ticker_count"] = len(data.get("tickers", {}))
    scores = []
    for payload in data.get("tickers", {}).values():
        qm = payload.get("quality_metrics", {}) if isinstance(payload, dict) else {}
        score = qm.get("quality_score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
    data["_meta"]["avg_quality_score"] = round(sum(scores) / len(scores), 4) if scores else 0.0
    
    with open(QUAL_FEATURES_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def is_stale(ticker_data: dict, stale_cutoff: datetime) -> bool:
    """Check if ticker data is stale."""
    prov = ticker_data.get('_provenance', {})
    last = prov.get('generated_at')
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(last.replace('Z', '+00:00'))
        # Remove timezone for comparison
        if last_dt.tzinfo:
            last_dt = last_dt.replace(tzinfo=None)
        return last_dt < stale_cutoff
    except Exception:
        return True

def is_valid_ticker_data(ticker_data: dict) -> bool:
    """Check if ticker data has all required fields with valid values."""
    if not ticker_data:
        return False
    
    # Must have sector and industry
    if not ticker_data.get('sector') or ticker_data.get('sector') == 'Unknown':
        return False
    if not ticker_data.get('industry') or ticker_data.get('industry') == 'Unknown':
        return False
    
    # Must have classifications
    cls = ticker_data.get('classifications', {})
    if not cls:
        return False
    
    # Check required classification fields exist
    required_cls = ['business_maturity', 'cyclical_exposure', 'moat_strength', 'debt_risk']
    for field in required_cls:
        if not cls.get(field):
            return False
    
    return True


def _stale_days_for_ticker(ticker: str, settings: dict) -> int:
    ticker_u = ticker.upper()
    if ticker_u in {t.upper() for t in PORTFOLIO_HOLDINGS}:
        return int(settings.get('stale_days_portfolio', STALE_DAYS_PORTFOLIO))
    if ticker_u in {t.upper() for t in WATCHLIST}:
        return int(settings.get('stale_days_watchlist', STALE_DAYS_WATCHLIST))
    return int(settings.get('stale_days_universe', STALE_DAYS_UNIVERSE))


def get_priority_tickers(existing_data: dict, all_tickers: list, max_count: int, settings: dict) -> list:
    """Return tickers that need updating, prioritized."""
    tickers_data = existing_data.get('tickers', {})
    now = datetime.now()
    
    priority = []
    
    # 0. Tickers with invalid/incomplete data (highest priority)
    for t in all_tickers:
        if t in tickers_data and not is_valid_ticker_data(tickers_data[t]):
            if t not in priority:
                priority.append(t)

    # 0b. Low-quality responses (requery sooner)
    for t in all_tickers:
        qm = (tickers_data.get(t, {}) or {}).get('quality_metrics', {})
        qscore = qm.get('quality_score', 1.0)
        if isinstance(qscore, (int, float)) and qscore < 0.55 and t not in priority:
            priority.append(t)
    
    # 1. Holdings not recently updated
    for t in PORTFOLIO_HOLDINGS:
        cutoff = now - timedelta(days=_stale_days_for_ticker(t, settings))
        if t not in tickers_data or is_stale(tickers_data.get(t, {}), cutoff):
            if t not in priority:
                priority.append(t)
    
    # 2. Watchlist not recently updated
    for t in WATCHLIST:
        cutoff = now - timedelta(days=_stale_days_for_ticker(t, settings))
        if t not in tickers_data or is_stale(tickers_data.get(t, {}), cutoff):
            if t not in priority:
                priority.append(t)
    
    # 3. Never analyzed
    for t in all_tickers:
        if t not in tickers_data and t not in priority:
            priority.append(t)
    
    # 4. Stale (old data)
    for t in all_tickers:
        cutoff = now - timedelta(days=_stale_days_for_ticker(t, settings))
        if t in tickers_data and is_stale(tickers_data[t], cutoff):
            if t not in priority:
                priority.append(t)
    
    return priority[:max_count]


def generate_prompt(ticker: str) -> str:
    """Generate prompt for single ticker with strict taxonomy."""
    return f'''Search online for current information about the ticker: {ticker}

You MUST search online and provide sources. Do not rely solely on training data.

Determine if this is an ETF or individual stock, then provide analysis.

Return ONLY valid JSON in this exact format:

FOR INDIVIDUAL STOCKS:
{{
  "ticker": "{ticker}",
  "is_etf": false,
  "sector": "MUST be exactly one of: Energy | Materials | Industrials | Consumer Discretionary | Consumer Staples | Healthcare | Financials | Technology | Communication Services | Utilities | Real Estate | Unknown",
  "industry": "MUST be exactly one of: Oil and Gas | Energy Equipment | Chemicals | Metals and Mining | Construction Materials | Aerospace and Defense | Industrial Machinery | Transportation | Retail | Automotive | Hotels and Leisure | Media and Entertainment | Food and Beverage | Household Products | Pharmaceuticals | Biotech | Healthcare Equipment | Healthcare Services | Banks | Insurance | Asset Management | Software | Hardware and Semiconductors | IT Services | Telecom | Utilities | REITs | Real Estate Services | Unknown",
  "business_description": "2-3 sentences describing what the company does",
  "classifications": {{
    "business_maturity": "startup | growth | mature | declining",
    "cyclical_exposure": "defensive | mixed | cyclical",
    "moat_strength": "none | narrow | wide",
    "debt_risk": "low | medium | high"
  }},
  "confidence": {{
    "sector": 0.0-1.0,
    "industry": 0.0-1.0,
    "business_maturity": 0.0-1.0,
    "cyclical_exposure": 0.0-1.0,
    "moat_strength": 0.0-1.0,
    "debt_risk": 0.0-1.0
  }},
  "key_risks": ["risk1", "risk2", "risk3"],
  "valuation_risk": "low | medium | high",
  "macro_sensitivity": {{
    "rates": "low | medium | high",
    "commodities": "low | medium | high",
    "fx": "low | medium | high"
  }},
  "catalysts_90d": ["catalyst1", "catalyst2"],
  "source_as_of_date": "YYYY-MM-DD",
  "sources": [
    {{"title": "Source name", "url": "https://..."}}
  ]
}}

FOR ETFs:
{{
  "ticker": "{ticker}",
  "is_etf": true,
  "sector": "Diversified (for broad market ETFs) OR the specific sector if it's a sector ETF",
  "industry": "MUST be exactly one of: Diversified ETF | Bond ETF | Sector ETF | International ETF | Unknown",
  "etf_type": "equity | bond | commodity | mixed",
  "etf_focus": "Brief description of ETF strategy/holdings",
  "holdings_summary": "Top 3-5 holdings or sector breakdown",
  "classifications": {{
    "business_maturity": "mature",
    "cyclical_exposure": "defensive | mixed | cyclical",
    "moat_strength": "none",
    "debt_risk": "low | medium | high"
  }},
  "confidence": {{
    "sector": 0.0-1.0,
    "industry": 0.0-1.0
  }},
  "key_risks": ["risk1", "risk2"],
  "valuation_risk": "low | medium | high",
  "macro_sensitivity": {{
    "rates": "low | medium | high",
    "commodities": "low | medium | high",
    "fx": "low | medium | high"
  }},
  "catalysts_90d": ["catalyst1", "catalyst2"],
  "source_as_of_date": "YYYY-MM-DD",
  "sources": [
    {{"title": "Source name", "url": "https://..."}}
  ]
}}

Classification Guidelines:
- business_maturity: startup=pre-profit/new, growth=expanding rapidly, mature=stable/established, declining=shrinking
- cyclical_exposure: defensive=utilities/staples/healthcare, mixed=moderate sensitivity, cyclical=highly sensitive
- moat_strength: none=commodity business, narrow=some advantage, wide=strong brand/patents/network effects
- debt_risk: low=net cash or minimal debt, medium=moderate leverage, high=highly leveraged

For ETFs:
- Broad market ETFs (SPY, VTI, etc.) → sector must be "Diversified", industry must be "Diversified ETF"
- Sector-specific ETFs (XLK, XLF, etc.) → sector = that sector, industry = "Sector ETF"
- Bond ETFs → sector = "Diversified", industry = "Bond ETF"
- moat_strength is always "none" for ETFs
- valuation_risk reflects valuation stretch risk (not business quality)
- macro_sensitivity should reflect primary sensitivities, not noise

CRITICAL:
- You MUST search online for current data
- You MUST provide at least one source
- Return ONLY the JSON object
- No markdown, no code blocks, no commentary
- If ticker not found, return: {{"ticker": "{ticker}", "error": "not_found"}}'''


def run_gemini_cli(prompt: str, timeout: int = 300, model: str = PRIMARY_MODEL) -> tuple:
    """Run gemini-cli with the given prompt. Returns (output, error_type).
    
    Args:
        prompt: The prompt to send to Gemini
        timeout: Timeout in seconds
        model: Model to use. Defaults to PRIMARY_MODEL (flash). Pass None for default gemini.
    """
    gemini_paths = [
        "gemini",
        "/opt/homebrew/bin/gemini",
        "/usr/local/bin/gemini",
        os.path.expanduser("~/.local/bin/gemini"),
    ]
    
    for gemini_cmd in gemini_paths:
        try:
            cmd = [gemini_cmd]
            if model:
                cmd.extend(["-m", model])
            cmd.extend(["-p", prompt])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            # Check for quota/rate limit errors in output
            # We check both stdout and stderr. 
            combined = (result.stdout + result.stderr)
            combined_lower = combined.lower()

            # 1. Hard Quota Detection (Daily limit / Resource Exhausted)
            # Only check if return code is non-zero OR if the output is very short (likely not a full response)
            if result.returncode != 0:
                hard_quota_keywords = [
                    'quota exceeded', 'quota limit', 'insufficient quota', 
                    'resource exhausted'
                ]
                if any(k in combined_lower for k in hard_quota_keywords):
                    # Extract reset time if available
                    reset_match = re.search(r'reset after\s+([0-9hHmMsS]+)', combined)
                    reset_msg = f" (Reset in {reset_match.group(1)})" if reset_match else ""
                    log_entry(f"[QUOTA] Detected hard quota (exit={result.returncode}){reset_msg}")
                    return combined, "daily_quota"

                # 2. Soft Rate Limit Detection (429 / Too Many Requests)
                rate_limit_keywords = [
                    'too many requests', '429 too many requests'
                ]
                if any(k in combined_lower for k in rate_limit_keywords):
                     reset_match = re.search(r'reset after\s+([0-9hHmMsS]+)', combined)
                     reset_msg = f" (Reset in {reset_match.group(1)})" if reset_match else ""
                     log_entry(f"[RATE_LIMIT] Detected rate limit (exit={result.returncode}){reset_msg}")
                     return combined, "rate_limit"

                # 3. HTTP 429 Detection (Contextual)
                # Look for "429" but ensure it's not part of a larger number (like 14290)
                # and is associated with "error" or "status" or stand-alone.
                if '429' in combined:
                    # Simple heuristic: if 'error' or 'status' is also present near '429'
                    if 'error' in combined_lower or 'status' in combined_lower or 'code' in combined_lower:
                         # Attempt to extract reset time even here, though less likely in standard 429
                         reset_match = re.search(r'reset after\s+([0-9hHmMsS]+)', combined)
                         reset_msg = f" (Reset in {reset_match.group(1)})" if reset_match else ""
                         log_entry(f"[RATE_LIMIT] Detected '429' with error context (exit={result.returncode}){reset_msg}")
                         return combined, "rate_limit"

            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip(), None
            
            # If we got here, it might be a silent failure or non-quota error
            if result.stderr:
                 # Return the stderr as the "output" for logging purposes if it fails
                 return result.stderr.strip(), "execution_error"
                 
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            print(f"[update_qual] Timeout after {timeout}s")
            return "", "timeout"
        except Exception as e:
            print(f"[update_qual] Error running {gemini_cmd}: {e}")
            return str(e), "execution_exception"
    
    print("[update_qual] ERROR: Could not find gemini-cli in any known location.")
    return "Gemini binary not found in standard paths", "not_found"


def repair_json_prompt(raw_output: str, ticker: str) -> str:
    """Generate a repair prompt for malformed JSON."""
    # Extract JSON-ish portion
    start = raw_output.find('{')
    end = raw_output.rfind('}') + 1
    if start >= 0 and end > start:
        raw_output = raw_output[start:end]
    
    return f'''The following output was supposed to be valid JSON for ticker {ticker}, but it has errors.
Fix the JSON and return ONLY the corrected JSON object. No commentary.

Original output:
{raw_output[:2000]}

Return ONLY valid JSON.'''


def validate_and_adjust(data: dict) -> dict:
    """Validate taxonomy and adjust confidence based on sources."""
    if not data or data.get('error'):
        return None
    
    # Check for required fields
    if 'sector' not in data or 'classifications' not in data:
        return None
    
    # Initialize confidence if missing
    if 'confidence' not in data:
        data['confidence'] = {}
    conf = data['confidence']
    if not isinstance(conf, dict):
        conf = {}
        data['confidence'] = conf

    # Clamp confidence values into [0, 1]
    for key, val in list(conf.items()):
        try:
            conf[key] = max(0.0, min(1.0, float(val)))
        except Exception:
            conf[key] = 0.0
    
    # Source validation - downgrade confidence if missing
    sources = data.get('sources', [])
    if not sources or len(sources) == 0:
        data['_unverified'] = True
        # Cap all confidence at 0.3
        for key in list(conf.keys()):
            if isinstance(conf[key], (int, float)):
                conf[key] = min(conf[key], 0.3)
    
    # Sector validation
    sector = data.get('sector', '')
    if sector not in VALID_SECTORS:
        # Try common variations
        sector_normalized = sector.title().strip()
        if sector_normalized in VALID_SECTORS:
            data['sector'] = sector_normalized
        else:
            data['sector'] = 'Unknown'
            conf['sector'] = 0.1
    
    # Industry validation
    industry = data.get('industry', '')
    if industry not in VALID_INDUSTRIES:
        # Try common variations
        industry_normalized = industry.title().strip()
        if industry_normalized in VALID_INDUSTRIES:
            data['industry'] = industry_normalized
        else:
            data['industry'] = 'Unknown'
            conf['industry'] = 0.1

    # Source as-of date sanity
    as_of = data.get('source_as_of_date')
    if as_of:
        try:
            parsed = datetime.fromisoformat(str(as_of).strip())
            data['source_as_of_date'] = parsed.strftime("%Y-%m-%d")
        except Exception:
            data['source_as_of_date'] = datetime.now().strftime("%Y-%m-%d")
    else:
        data['source_as_of_date'] = datetime.now().strftime("%Y-%m-%d")
    
    # Classifications validation
    cls = data.get('classifications', {})
    if not isinstance(cls, dict):
        cls = {}
        data['classifications'] = cls
    if cls.get('business_maturity') not in VALID_MATURITY:
        cls['business_maturity'] = 'mature'
    if cls.get('cyclical_exposure') not in VALID_CYCLICAL:
        cls['cyclical_exposure'] = 'mixed'
    if cls.get('moat_strength') not in VALID_MOAT:
        cls['moat_strength'] = 'narrow' if not data.get('is_etf') else 'none'
    if cls.get('debt_risk') not in VALID_DEBT:
        cls['debt_risk'] = 'medium'

    # New optional fields: valuation and macro sensitivity
    valuation_risk = str(data.get('valuation_risk', 'medium')).strip().lower()
    if valuation_risk not in VALID_VALUATION_RISK:
        valuation_risk = 'medium'
    data['valuation_risk'] = valuation_risk

    macro = data.get('macro_sensitivity', {})
    if not isinstance(macro, dict):
        macro = {}
    rates = str(macro.get('rates', 'medium')).strip().lower()
    commodities = str(macro.get('commodities', 'medium')).strip().lower()
    fx = str(macro.get('fx', 'medium')).strip().lower()
    if rates not in VALID_RATES_SENSITIVITY:
        rates = 'medium'
    if commodities not in VALID_COMMODITY_SENSITIVITY:
        commodities = 'medium'
    if fx not in VALID_FX_SENSITIVITY:
        fx = 'medium'
    data['macro_sensitivity'] = {
        'rates': rates,
        'commodities': commodities,
        'fx': fx,
    }

    if not isinstance(data.get('catalysts_90d'), list):
        data['catalysts_90d'] = []
    data['catalysts_90d'] = [str(x).strip() for x in data.get('catalysts_90d', []) if str(x).strip()][:5]

    # Quality metrics for downstream filtering/audits
    conf_values = [float(v) for v in conf.values() if isinstance(v, (int, float))]
    mean_conf = (sum(conf_values) / len(conf_values)) if conf_values else 0.0

    required_checks = [
        data.get('sector') not in (None, '', 'Unknown'),
        data.get('industry') not in (None, '', 'Unknown'),
        bool(cls.get('business_maturity')),
        bool(cls.get('cyclical_exposure')),
        bool(cls.get('moat_strength')),
        bool(cls.get('debt_risk')),
        bool(data.get('valuation_risk')),
        bool(data.get('macro_sensitivity', {}).get('rates')),
    ]
    completeness = sum(1 for ok in required_checks if ok) / max(len(required_checks), 1)
    source_count = len(sources) if isinstance(sources, list) else 0

    quality_score = (
        0.50 * mean_conf
        + 0.30 * completeness
        + 0.20 * min(source_count / 3.0, 1.0)
    )

    data['quality_metrics'] = {
        'source_count': int(source_count),
        'mean_confidence': float(round(mean_conf, 4)),
        'completeness': float(round(completeness, 4)),
        'quality_score': float(round(quality_score, 4)),
    }
    
    # Add provenance
    data['_provenance'] = {
        'generated_at': datetime.now().isoformat(),
        'prompt_version': 'v4'
    }
    
    return data


def parse_response(response: str) -> dict:
    """Parse gemini-cli JSON response."""
    if not response:
        return None
    
    # Strip markdown fences if present
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        response = "\n".join(lines)
    
    # Try to extract JSON portion
    start = response.find('{')
    end = response.rfind('}') + 1
    if start >= 0 and end > start:
        response = response[start:end]
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None


def process_ticker(ticker: str, max_retries: int = 2, use_fallback_model: bool = False) -> tuple:
    """Process a single ticker with retry logic. Returns (data, error_reason, used_fallback).
    
    Args:
        ticker: The ticker symbol to process
        max_retries: Number of retry attempts
        use_fallback_model: If True, use the fallback model (default gemini, no -m flag)
    
    Returns:
        (data, error_reason, used_fallback) - used_fallback indicates if fallback was triggered
    """
    print(f"[update_qual] Processing {ticker}...")
    
    prompt = generate_prompt(ticker)
    last_raw_response = ""
    used_fallback = use_fallback_model
    
    # Determine which model to use: PRIMARY_MODEL (flash) by default, FALLBACK_MODEL (None/default) as fallback
    model_to_use = FALLBACK_MODEL if use_fallback_model else PRIMARY_MODEL
    if use_fallback_model:
        print(f"[update_qual] Using fallback model (default gemini)")
    
    for attempt in range(max_retries):
        response, error = run_gemini_cli(prompt, model=model_to_use)
        last_raw_response = response

        if error == "daily_quota":
            return None, "daily_quota", used_fallback
        
        if error == "rate_limit" and not use_fallback_model:
            # Try fallback model (default gemini) once
            log_entry(f"[FALLBACK] Rate limit on primary model ({PRIMARY_MODEL}) for {ticker}. Switching to default gemini...")
            print(f"[update_qual] Rate limit hit. Switching to fallback model (default gemini)")
            used_fallback = True
            model_to_use = FALLBACK_MODEL  # None = default gemini
            response, error = run_gemini_cli(prompt, model=FALLBACK_MODEL)
            last_raw_response = response
            
            if error == "daily_quota":
                return None, "daily_quota", used_fallback
            if error == "rate_limit":
                log_entry(f"[FAIL] Fallback model (default gemini) also rate limited for {ticker}. Stopping.")
                return None, "rate_limit_all_models", used_fallback
            # If fallback succeeded (no error), continue to parsing below
        elif error == "rate_limit" and use_fallback_model:
            # Already on fallback (default gemini) and still rate limited
            log_entry(f"[FAIL] Fallback model (default gemini) rate limited for {ticker}. Stopping.")
            return None, "rate_limit_all_models", used_fallback

        if not error:
            data = parse_response(response)
            if data:
                validated = validate_and_adjust(data)
                if validated:
                    qscore = (validated.get('quality_metrics') or {}).get('quality_score', 0.0)
                    print(
                        f"[update_qual] {ticker}: SUCCESS "
                        f"(sector={validated.get('sector')}, industry={validated.get('industry')}, q={qscore:.2f})"
                    )
                    return validated, None, used_fallback
            
            # If parsing failed, try repair
            if response and attempt < max_retries - 1:
                print(f"[update_qual] {ticker}: Attempting JSON repair...")
                repair_prompt = repair_json_prompt(response, ticker)
                response, error = run_gemini_cli(repair_prompt, timeout=60, model=model_to_use)
                
                if error == "daily_quota": return None, "daily_quota", used_fallback
                if error == "rate_limit":
                    return None, "rate_limit_during_repair", used_fallback

                data = parse_response(response)
                if data:
                    validated = validate_and_adjust(data)
                    if validated:
                        print(f"[update_qual] {ticker}: SUCCESS after repair")
                        return validated, None, used_fallback
        
        time.sleep(1)
    
    print(f"[update_qual] {ticker}: FAILED after {max_retries} attempts")
    snippet = last_raw_response[:50].replace('\n', ' ') if last_raw_response else "No response"
    return None, f"parse_failed[{snippet}...]", used_fallback


def main():
    parser = argparse.ArgumentParser(description="Update qualitative features via gemini-cli")
    parser.add_argument('--ticker', type=str, help='Process single ticker')
    parser.add_argument('--count', type=int, help='Override chunk size')
    parser.add_argument('--list-priority', action='store_true', help='List priority tickers without processing')
    args = parser.parse_args()
    
    # Load settings
    settings = load_settings()
    chunk_size = args.count or settings.get('chunk_size', DEFAULT_CHUNK_SIZE)
    
    # Load existing data
    existing = load_existing_data()
    
    if args.ticker:
        # Single ticker mode
        result, error, _ = process_ticker(args.ticker.upper())
        total = len(existing.get('tickers', {}))
        if result:
            existing['tickers'][args.ticker.upper()] = result
            save_data(existing)
            total = len(existing.get('tickers', {}))
            log_entry(f"[OK] {args.ticker.upper()}", total)
            print(f"[update_qual] Saved {args.ticker.upper()}")
        else:
            log_entry(f"[FAIL] {args.ticker.upper()}: {error or 'unknown'}", total)
        return
    
    # Get all tickers and determine priority
    all_tickers = get_all_tickers()
    priority = get_priority_tickers(existing, all_tickers, chunk_size, settings)
    
    if args.list_priority:
        print(f"[update_qual] Priority tickers (next {chunk_size}):")
        for t in priority:
            print(f"  - {t}")
        return
    
    if not priority:
        log_entry("[SKIP] All tickers up to date", len(existing.get('tickers', {})))
        print("[update_qual] All tickers up to date!")
        return
    
    print(f"[update_qual] Processing {len(priority)} tickers...")
    print(
        "[update_qual] Freshness windows: "
        f"holdings={settings.get('stale_days_portfolio', STALE_DAYS_PORTFOLIO)}d, "
        f"watchlist={settings.get('stale_days_watchlist', STALE_DAYS_WATCHLIST)}d, "
        f"universe={settings.get('stale_days_universe', STALE_DAYS_UNIVERSE)}d"
    )
    
    success_count = 0
    fail_count = 0
    quota_hit = False
    use_fallback = False  # Track if we should use fallback model
    prompt_count = 0  # Track prompts for auto-switch
    
    log_entry(f"[START] Processing {len(priority)} tickers (primary={PRIMARY_MODEL})", len(existing.get('tickers', {})))
    
    for i, ticker in enumerate(priority):
        print(f"\n[{i+1}/{len(priority)}] ", end="")
        
        # Auto-switch to fallback after AUTO_SWITCH_PROMPTS prompts
        if not use_fallback and prompt_count >= AUTO_SWITCH_PROMPTS:
            log_entry(f"[AUTO_SWITCH] Switching to fallback (default gemini) after {prompt_count} prompts", len(existing.get('tickers', {})))
            print(f"[update_qual] Auto-switching to fallback model after {prompt_count} prompts")
            use_fallback = True
        
        result, error, used_fallback = process_ticker(ticker, use_fallback_model=use_fallback)
        prompt_count += 1  # Count each prompt attempt
        total = len(existing.get('tickers', {}))
        
        # If fallback was triggered by rate limit, use it for all remaining tickers
        if used_fallback and not use_fallback:
            log_entry(f"[MODEL_SWITCH] Switching to fallback (default gemini) for remaining tickers", total)
            use_fallback = True
        
        if result:
            existing['tickers'][ticker] = result
            success_count += 1
            consecutive_failures = 0  # Reset on success
            # Save after each successful ticker (resume capability)
            save_data(existing)
            total = len(existing.get('tickers', {}))
            log_entry(f"[OK] {ticker} ({i+1}/{len(priority)})", total)
        else:
            fail_count += 1
            consecutive_failures += 1
            # Clean up error message for log
            clean_error = str(error).replace('\n', ' ')[:100]
            log_entry(f"[FAIL] {ticker}: {clean_error} ({i+1}/{len(priority)})", total)
            
            # Stop on quota errors
            if error == "daily_quota":
                quota_hit = True
                log_entry(f"[STOP] Quota limit reached after {i+1} tickers", total)
                break
            
            # Stop if both models hit rate limit
            if error == "rate_limit_all_models":
                log_entry(f"[STOP] All models rate limited for {ticker}. Stopping.", total)
                print(f"\n[update_qual] STOPPING: All models rate limited.")
                break
            
            # Stop on ANY failure (Strict Mode requested by user)
            log_entry(f"[STOP] Strict mode: stopping after failure on {ticker}", total)
            print(f"\n[update_qual] STOPPING: Failed to process {ticker}. strict failure mode.")
            break
        
        # Delay between calls to respect quotas
        if i < len(priority) - 1:
            time.sleep(DELAY_BETWEEN_CALLS)
    
    summary = f"[DONE] {success_count} success, {fail_count} failed"
    if quota_hit:
        summary += " (stopped: quota)"
    log_entry(summary, len(existing.get('tickers', {})))
    print(f"\n[update_qual] Complete: {success_count} success, {fail_count} failed")



if __name__ == "__main__":
    main()
