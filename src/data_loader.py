
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import random
import time
import json
import os
import re
import io
import shutil
import subprocess

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1"
]

try:
    from src.rate_limiter import TokenBucket, APICircuitBreaker
except ImportError:
    from rate_limiter import TokenBucket, APICircuitBreaker


def _load_alpha_vantage_keys() -> list[str]:
    """Load configured Alpha Vantage keys, supporting multi-key rotation."""
    try:
        from config import ALPHA_VANTAGE_API_KEYS, ALPHA_VANTAGE_API_KEY
        keys = [str(k).strip() for k in (ALPHA_VANTAGE_API_KEYS or []) if str(k).strip()]
        if not keys and ALPHA_VANTAGE_API_KEY:
            keys = [str(ALPHA_VANTAGE_API_KEY).strip()]
        return keys
    except Exception:
        return []


_ALPHA_VANTAGE_KEYS = _load_alpha_vantage_keys()
_ALPHAVANTAGE_KEYS_COUNT = max(1, len(_ALPHA_VANTAGE_KEYS))
_ALPHAVANTAGE_KEY_INDEX = 0
_ALPHAVANTAGE_KEY_LOCK = __import__('threading').Lock()


def _next_alpha_vantage_key() -> str:
    """Return next Alpha Vantage key in round-robin order."""
    global _ALPHAVANTAGE_KEY_INDEX
    if not _ALPHA_VANTAGE_KEYS:
        return ""
    with _ALPHAVANTAGE_KEY_LOCK:
        key = _ALPHA_VANTAGE_KEYS[_ALPHAVANTAGE_KEY_INDEX % len(_ALPHA_VANTAGE_KEYS)]
        _ALPHAVANTAGE_KEY_INDEX += 1
    return key

# =============================================================================
# RATE LIMITER & CIRCUIT BREAKER
# =============================================================================

# Global limiters
# Finnhub: 60 calls/min = 1/sec
_LIMITER_FINNHUB = TokenBucket(tokens=60, fill_rate=1.0)
_CB_FINNHUB = APICircuitBreaker("Finnhub", failure_threshold=5, recovery_timeout=300)

# Twelve Data: API returns 429 "Run out of credits" if >8/min
# Confirmed via logs: "9 API credits were used... limit reached"
_LIMITER_TWELVEDATA = TokenBucket(tokens=8, fill_rate=8.0/60.0)
_CB_TWELVEDATA = APICircuitBreaker("Twelve Data", failure_threshold=3, recovery_timeout=300)

# Alpaca: 200 calls/min = 3.33/sec (Paper Trading)
_LIMITER_ALPACA = TokenBucket(tokens=200, fill_rate=200.0/60.0)
_CB_ALPACA = APICircuitBreaker("Alpaca", failure_threshold=5, recovery_timeout=60)

# Tiingo: 500 calls/hour ~ 8.3/min ~ 0.14/sec
# Allow bursts up to token limit (e.g., 50)
_LIMITER_TIINGO = TokenBucket(tokens=50, fill_rate=500.0/3600.0)
_CB_TIINGO = APICircuitBreaker("Tiingo", failure_threshold=5, recovery_timeout=300)

# Polygon: 5 calls/min = 0.083/sec
_LIMITER_POLYGON = TokenBucket(tokens=5, fill_rate=5.0/60.0)
_CB_POLYGON = APICircuitBreaker("Polygon", failure_threshold=3, recovery_timeout=300)

# Alpha Vantage: ~5 calls/min per key (scale with configured key count)
_LIMITER_ALPHAVANTAGE = TokenBucket(
    tokens=5 * _ALPHAVANTAGE_KEYS_COUNT,
    fill_rate=(5.0 * _ALPHAVANTAGE_KEYS_COUNT) / 60.0,
)
_CB_ALPHAVANTAGE = APICircuitBreaker("Alpha Vantage", failure_threshold=3, recovery_timeout=300)

# Financial Modeling Prep (free tiers vary; keep conservative by default)
_LIMITER_FMP = TokenBucket(tokens=5, fill_rate=5.0/60.0)
_CB_FMP = APICircuitBreaker("FMP", failure_threshold=3, recovery_timeout=300)

# EODHD (free tiers vary; keep conservative by default)
_LIMITER_EODHD = TokenBucket(tokens=5, fill_rate=5.0/60.0)
_CB_EODHD = APICircuitBreaker("EODHD", failure_threshold=3, recovery_timeout=300)

# yfinance: ~100 calls/min is safe before throttling
_LIMITER_YFINANCE = TokenBucket(tokens=100, fill_rate=100.0/60.0)
_CB_YFINANCE = APICircuitBreaker("yfinance", failure_threshold=5, recovery_timeout=60)

# Stooq (No strict limits, but use CB for soft failures)
_CB_STOOQ = APICircuitBreaker("Stooq", failure_threshold=10, recovery_timeout=60)

# Gemini CLI last-ditch fallback (very expensive/slow; strict breaker)
_CB_GEMINI = APICircuitBreaker("Gemini CLI", failure_threshold=2, recovery_timeout=600)

# Session cache for connection pooling
_SESSION_CACHE = {}
_SESSION_LOCK = __import__('threading').Lock()

# API Performance Metrics
_API_STATS = {
    'yfinance': {'calls': 0, 'success': 0, 'failures': 0, 'total_time': 0.0},
    'alpaca': {'calls': 0, 'success': 0, 'failures': 0, 'total_time': 0.0},
    'tiingo': {'calls': 0, 'success': 0, 'failures': 0, 'total_time': 0.0},
    'stooq': {'calls': 0, 'success': 0, 'failures': 0, 'total_time': 0.0},
    'twelve_data': {'calls': 0, 'success': 0, 'failures': 0, 'total_time': 0.0},
    'finnhub': {'calls': 0, 'success': 0, 'failures': 0, 'total_time': 0.0},
    'polygon': {'calls': 0, 'success': 0, 'failures': 0, 'total_time': 0.0},
    'alpha_vantage': {'calls': 0, 'success': 0, 'failures': 0, 'total_time': 0.0},
    'fmp': {'calls': 0, 'success': 0, 'failures': 0, 'total_time': 0.0},
    'eodhd': {'calls': 0, 'success': 0, 'failures': 0, 'total_time': 0.0},
    'gemini_cli': {'calls': 0, 'success': 0, 'failures': 0, 'total_time': 0.0},
}
_STATS_LOCK = __import__('threading').Lock()

def _log_api_call(api_name: str, duration: float, success: bool, ticker: str = ""):
    """Log API call metrics for performance monitoring."""
    with _STATS_LOCK:
        _API_STATS[api_name]['calls'] += 1
        _API_STATS[api_name]['total_time'] += duration
        if success:
            _API_STATS[api_name]['success'] += 1
        else:
            _API_STATS[api_name]['failures'] += 1
    
    status = "OK" if success else "FAIL"
    avg_time = _API_STATS[api_name]['total_time'] / max(_API_STATS[api_name]['calls'], 1)
    print(f"  [{api_name}] {status} {ticker} in {duration:.2f}s (avg: {avg_time:.2f}s, success: {_API_STATS[api_name]['success']}/{_API_STATS[api_name]['calls']})", flush=True)

def get_api_stats():
    """Return current API performance statistics."""
    return dict(_API_STATS)

def _create_session():
    """Create a robust requests session with retries and SSL adapter."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    session = requests.Session()
    # Disable SSL verification for M1 Mac compatibility (Broken Pipe/SSL Bad Record Mac)
    session.verify = False 
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    retry = Retry(
        total=3,
        read=3,
        connect=3,
        backoff_factor=0.5,
        status_forcelist=(500, 502, 503, 504, 429),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def _get_session(api_name="default"):
    """Get a cached session for connection pooling."""
    with _SESSION_LOCK:
        if api_name not in _SESSION_CACHE:
            _SESSION_CACHE[api_name] = _create_session()
        return _SESSION_CACHE[api_name]

def _is_rate_limit_error(error: Exception) -> bool:
    """Detect if an exception indicates rate limiting."""
    err_str = str(error).lower()
    return any(x in err_str for x in ['429', 'rate limit', 'too many requests', 'throttle'])


def _normalize_ohlcv_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize downloaded OHLCV data into the project schema."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]

    if 'datetime' in out.columns and 'date' not in out.columns:
        out = out.rename(columns={'datetime': 'date'})
    if 'timestamp' in out.columns and 'date' not in out.columns:
        out = out.rename(columns={'timestamp': 'date'})

    required = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in out.columns for col in required):
        return pd.DataFrame()

    out = out[required].copy()
    out['date'] = pd.to_datetime(out['date'], errors='coerce')
    out = out.dropna(subset=['date'])
    if out.empty:
        return pd.DataFrame()

    for col in ['open', 'high', 'low', 'close', 'volume']:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out = out.dropna(subset=['open', 'high', 'low', 'close'])
    if out.empty:
        return pd.DataFrame()

    out['ticker'] = ticker
    out = out.sort_values('date').drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)
    return out[['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']]


def _extract_json_blob(text: str) -> str:
    """Extract JSON-like payload from CLI output."""
    raw = (text or "").strip()
    if not raw:
        return ""
    if raw.startswith("```"):
        lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start:end + 1]
    return raw


def _is_plausible_ticker(ticker: str) -> bool:
    """Basic symbol sanity check to avoid wasting expensive fallbacks."""
    symbol = (ticker or "").strip().upper()
    return bool(re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", symbol))


def _run_gemini_cli_prompt(prompt: str, model: str = "gemini-3-flash-preview", timeout: int = 180) -> tuple[str, str | None]:
    """Run gemini-cli prompt. Returns (stdout, error_tag)."""
    gemini_cmd = (
        shutil.which("gemini")
        or shutil.which("gemini-cli")
        or "/opt/homebrew/bin/gemini"
        or "/usr/local/bin/gemini"
        or os.path.expanduser("~/.local/bin/gemini")
    )
    if not gemini_cmd or not os.path.exists(gemini_cmd):
        return "", "not_found"

    cmd = [gemini_cmd]
    if model:
        cmd.extend(["-m", model])
    cmd.extend(["-p", prompt])

    try:
        started = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        duration = time.time() - started
        combined = f"{proc.stdout}\n{proc.stderr}".lower()
        if proc.returncode == 0 and proc.stdout.strip():
            _log_api_call("gemini_cli", duration, True, "batch")
            return proc.stdout.strip(), None
        if "quota" in combined or "resource exhausted" in combined:
            _log_api_call("gemini_cli", duration, False, "batch_quota")
            return proc.stdout.strip() or proc.stderr.strip(), "quota"
        if "429" in combined or "too many requests" in combined or "rate limit" in combined:
            _log_api_call("gemini_cli", duration, False, "batch_rl")
            return proc.stdout.strip() or proc.stderr.strip(), "rate_limit"
        _log_api_call("gemini_cli", duration, False, "batch_err")
        return proc.stdout.strip() or proc.stderr.strip(), "execution_error"
    except subprocess.TimeoutExpired:
        return "", "timeout"
    except Exception:
        return "", "execution_exception"


def _parse_gemini_price_payload(raw_output: str) -> dict:
    """Parse gemini-cli JSON payload for batched OHLCV results."""
    blob = _extract_json_blob(raw_output)
    if not blob:
        return {}
    try:
        data = json.loads(blob)
    except Exception:
        return {}

    items = data.get("tickers", [])
    if isinstance(data, dict) and not items and isinstance(data.get("data"), list):
        items = data.get("data")
    parsed = {}
    for item in items:
        ticker = str(item.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        bars = item.get("bars", [])
        if not isinstance(bars, list):
            continue
        parsed[ticker] = bars
    return parsed

def split_and_download_ticker(ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Downloads historical market data for a single ticker using yfinance."""
    # Check rate limiter first
    if not _LIMITER_YFINANCE.consume():
        print(f"  [yfinance] Rate limited, skipping {ticker}")
        return pd.DataFrame()
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')

    start_time = time.time()
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False, threads=False)
        duration = time.time() - start_time
        
        if df.empty:
            _log_api_call('yfinance', duration, False, ticker)
            return pd.DataFrame()

        df = df.reset_index()
        
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).lower() for c in df.columns]
        df['ticker'] = ticker
        
        _log_api_call('yfinance', duration, True, ticker)
        return df

    except Exception as e:
        duration = time.time() - start_time
        _log_api_call('yfinance', duration, False, ticker)
        print(f"Error downloading {ticker}: {e}", flush=True)
        return pd.DataFrame()

def download_batch(tickers: list, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Downloads historical market data for multiple tickers efficiently.
    Uses exponential backoff with jitter for rate limit handling.
    """
    if not tickers:
        return pd.DataFrame()
    
    # Check rate limiter - consume tokens proportional to batch size
    # This prevents batch downloads from bypassing rate limiting
    tokens_needed = min(len(tickers), 20)  # Cap at 20 to avoid depleting bucket
    if not _LIMITER_YFINANCE.consume(tokens_needed):
        print(f"  [yfinance-batch] Rate limited ({tokens_needed} tokens needed), skipping batch")
        return pd.DataFrame()
        
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')

    print(f"[yfinance-batch] Downloading {len(tickers)} tickers...", flush=True)
    start_time = time.time()
    
    try:
        df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False, group_by='ticker', threads=False)
        duration = time.time() - start_time
        
        if df.empty:
            print(f"[yfinance-batch] ??? Empty result in {duration:.2f}s", flush=True)
            return pd.DataFrame()

        long_df = df.stack(level=0).reset_index()
        long_df.columns = [str(c).lower() for c in long_df.columns]
        
        if 'level_1' in long_df.columns:
            long_df = long_df.rename(columns={'level_1': 'ticker'})
        
        success_count = long_df['ticker'].nunique() if 'ticker' in long_df.columns else 0
        print(f"[yfinance-batch] ??? Got {success_count}/{len(tickers)} tickers in {duration:.2f}s ({duration/len(tickers):.2f}s/ticker)", flush=True)
        return long_df

    except Exception as e:
        duration = time.time() - start_time
        print(f"[yfinance-batch] ??? Error in {duration:.2f}s: {e}", flush=True)
        return pd.DataFrame()

def download_from_stooq(ticker: str) -> pd.DataFrame:
    """Fallback downloader using Stooq (CSV export)."""
    if not _CB_STOOQ.allow_request():
        return pd.DataFrame()

    url = f"https://stooq.com/q/d/l/?s={ticker}.us&i=d"
    start_time = time.time()
    try:
        # Use requests to check content first
        session = _get_session("stooq")
        resp = session.get(url, timeout=10)
        duration = time.time() - start_time
        
        if resp.status_code != 200:
             _log_api_call('stooq', duration, False, ticker)
             return pd.DataFrame()
             
        content = resp.text
        if "Exceeded the daily hits limit" in content or "<!DOCTYPE html>" in content:
             print(f"  [stooq] Blocked/Rate Limited for {ticker}", flush=True)
             _CB_STOOQ.record_failure() # Trip CB to stop hammering
             _log_api_call('stooq', duration, False, ticker)
             return pd.DataFrame()

        from io import StringIO
        df = pd.read_csv(StringIO(content))
        
        if df.empty or len(df) < 10:
            _log_api_call('stooq', duration, False, ticker)
            return pd.DataFrame()
        
        _CB_STOOQ.record_success()
        _log_api_call('stooq', duration, True, ticker)
        
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={'vol': 'volume'})
        df['ticker'] = ticker
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        duration = time.time() - start_time
        _CB_STOOQ.record_failure()
        _log_api_call('stooq', duration, False, ticker)
        print(f"  [stooq] Error {ticker}: {str(e)[:50]}", flush=True)
        return pd.DataFrame()

def download_from_alpha_vantage(ticker: str) -> pd.DataFrame:
    """Fallback downloader using Alpha Vantage API."""
    if not _CB_ALPHAVANTAGE.allow_request():
        return pd.DataFrame()

    if not _ALPHA_VANTAGE_KEYS:
        return pd.DataFrame()

    print(f"Attempting Alpha Vantage fallback for {ticker}...")

    # TIME_SERIES_DAILY is available on free tier; DAILY_ADJUSTED may require premium.
    session = _get_session("alpha_vantage")
    started = time.time()
    attempts = 0

    for _ in range(len(_ALPHA_VANTAGE_KEYS)):
        if not _LIMITER_ALPHAVANTAGE.consume():
            print(f"  Alpha Vantage rate limited (TokenBucket), skipping {ticker}")
            break

        api_key = _next_alpha_vantage_key()
        if not api_key:
            break
        attempts += 1
        url = (
            "https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=compact&datatype=csv&apikey={api_key}"
        )

        try:
            resp = session.get(url, timeout=12)
            if resp.status_code != 200:
                continue

            body = (resp.text or "").strip()
            # Quota/limit/errors come back as JSON or plain text instead of CSV.
            if body.startswith("{") or "Thank you for using Alpha Vantage" in body:
                continue

            df = pd.read_csv(io.StringIO(body))
            if df.empty or 'timestamp' not in df.columns:
                continue

            # Normalize column names for free-tier DAILY endpoint.
            df = df.rename(columns={'timestamp': 'date'})
            df.columns = [c.lower() for c in df.columns]
            df['ticker'] = ticker
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            duration = time.time() - started
            _CB_ALPHAVANTAGE.record_success()
            _log_api_call('alpha_vantage', duration, True, ticker)
            return df[['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']]
        except Exception:
            continue

    duration = time.time() - started
    _CB_ALPHAVANTAGE.record_failure()
    _log_api_call('alpha_vantage', duration, False, f"{ticker} (attempts={attempts})")
    return pd.DataFrame()

def download_from_finnhub(ticker: str) -> pd.DataFrame:
    """Fallback downloader using Finnhub API (60 calls/min free tier)."""
    if not _CB_FINNHUB.allow_request():
        return pd.DataFrame()
        
    if not _LIMITER_FINNHUB.consume():
        print(f"  Finnhub rate limited (TokenBucket), skipping {ticker}")
        return pd.DataFrame()

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import FINNHUB_API_KEY
    
    if not FINNHUB_API_KEY:
        return pd.DataFrame()  # Skip if no API key configured
    
    end_ts = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=365*2)).timestamp())
    url = f"https://finnhub.io/api/v1/stock/candle?symbol={ticker}&resolution=D&from={start_ts}&to={end_ts}&token={FINNHUB_API_KEY}"
    
    start_time = time.time()
    try:
        session = _get_session("finnhub")
        resp = session.get(url, timeout=10)
        duration = time.time() - start_time
        
        if resp.status_code == 429:
            print(f"  [finnhub] 429 Rate Limit {ticker}", flush=True)
            _log_api_call('finnhub', duration, False, ticker)
            return pd.DataFrame()
        
        data = resp.json()
        if data.get('s') != 'ok' or 'c' not in data:
            _log_api_call('finnhub', duration, False, ticker)
            return pd.DataFrame()
        
        _CB_FINNHUB.record_success()
        _log_api_call('finnhub', duration, True, ticker)
        
        df = pd.DataFrame({
            'date': pd.to_datetime(data['t'], unit='s'),
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c'],
            'volume': data['v'],
            'ticker': ticker
        })
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        duration = time.time() - start_time
        _CB_FINNHUB.record_failure()
        _log_api_call('finnhub', duration, False, ticker)
        print(f"  [finnhub] Error {ticker}: {str(e)[:50]}", flush=True)
        return pd.DataFrame()

def download_from_polygon(ticker: str) -> pd.DataFrame:
    """Fallback downloader using Polygon.io API (5 calls/min free tier)."""
    if not _CB_POLYGON.allow_request():
        return pd.DataFrame()
        
    if not _LIMITER_POLYGON.consume():
        print(f"  Polygon rate limited (TokenBucket), skipping {ticker}")
        return pd.DataFrame()

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import POLYGON_API_KEY
    
    if not POLYGON_API_KEY:
        return pd.DataFrame()  # Skip if no API key configured
    
    print(f"  Attempting Polygon.io fallback for {ticker}...")
    
    try:
        # Get 2 years of daily data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={POLYGON_API_KEY}&limit=50000"
        
        session = _get_session("polygon")
        resp = session.get(url, timeout=10)
        
        if resp.status_code == 429:
            print("  Polygon 429 Rate Limit Hit.")
            return pd.DataFrame()
        
        data = resp.json()
        # Some responses omit 'status' while still returning valid 'results'.
        if 'results' not in data or not data.get('results'):
            return pd.DataFrame()
        
        _CB_POLYGON.record_success()

        results = data['results']
        df = pd.DataFrame({
            'date': pd.to_datetime([r['t'] for r in results], unit='ms'),
            'open': [r['o'] for r in results],
            'high': [r['h'] for r in results],
            'low': [r['l'] for r in results],
            'close': [r['c'] for r in results],
            'volume': [r['v'] for r in results],
            'ticker': ticker
        })
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        _CB_POLYGON.record_failure()
        print(f"  Polygon.io download failed for {ticker}: {e}")
        return pd.DataFrame()

def download_from_alpaca(ticker: str) -> pd.DataFrame:
    """Fallback downloader using Alpaca Data API (High Speed)."""
    print(f"  [debug-alpaca] Entering for {ticker}", flush=True)
    if not _CB_ALPACA.allow_request():
        print(f"  [debug-alpaca] CB Open for {ticker}", flush=True)
        return pd.DataFrame()
        
    if not _LIMITER_ALPACA.consume():
        print(f"  [alpaca] Rate limited, skipping {ticker}", flush=True)
        return pd.DataFrame()

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
    
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return pd.DataFrame()

    # Alpaca V2 Bars Endpoint (free data requires feed=iex)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    url = (
        f"{ALPACA_BASE_URL}/stocks/{ticker}/bars"
        f"?timeframe=1Day&limit=10000&adjustment=all"
        f"&start={start_date}&end={end_date}&feed=iex"
    )
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    
    start_time = time.time()
    try:
        session = _get_session("alpaca")
        # Disable SSL verify for M1 Mac compatibility
        resp = session.get(url, headers=headers, timeout=10, verify=False)
        duration = time.time() - start_time
        
        if resp.status_code != 200:
             print(f"  [alpaca] HTTP {resp.status_code} for {ticker}: {resp.text[:200]}", flush=True)
             _log_api_call('alpaca', duration, False, ticker)
             if resp.status_code in [401, 403]:
                 _CB_ALPACA.record_failure()
             return pd.DataFrame()

        data = resp.json()
        if "bars" not in data or not data["bars"]:
             print(f"  [alpaca] Empty bars for {ticker}: {resp.text[:200]}", flush=True)
             _log_api_call('alpaca', duration, False, ticker)
             return pd.DataFrame()

        _CB_ALPACA.record_success()
        _log_api_call('alpaca', duration, True, ticker)
        
        # Parse bars: t=time, o=open, h=high, l=low, c=close, v=volume
        bars = data["bars"]
        df = pd.DataFrame(bars)
        df = df.rename(columns={
            't': 'date', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
        })
        # Drop other columns like 'n', 'vw'
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        
        df['ticker'] = ticker
        df['date'] = pd.to_datetime(df['date']).dt.tz_convert(None) # Remove timezone
        df = df.sort_values('date').reset_index(drop=True)
        return df

    except Exception as e:
        duration = time.time() - start_time
        _CB_ALPACA.record_failure()
        _log_api_call('alpaca', duration, False, ticker)
        print(f"  [alpaca] Error {ticker}: {str(e)[:50]}", flush=True)
        return pd.DataFrame()

def download_from_tiingo(ticker: str) -> pd.DataFrame:
    """Fallback downloader using Tiingo API."""
    if not _CB_TIINGO.allow_request():
        return pd.DataFrame()
        
    if not _LIMITER_TIINGO.consume():
        print(f"  [tiingo] Rate limited, skipping {ticker}", flush=True)
        return pd.DataFrame()

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import TIINGO_API_KEY
    
    if not TIINGO_API_KEY:
        return pd.DataFrame()

    # Tiingo EOD Endpoint
    # format=json is default
    # startDate handled by default (returns all available or scoped)
    # We want last 5 years usually, but let's grab default first
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?token={TIINGO_API_KEY}&startDate={start_date}"
    
    start_time = time.time()
    try:
        session = _get_session("tiingo")
        headers = {'Content-Type': 'application/json'}
        # Disable SSL verify for M1 Mac compatibility
        resp = session.get(url, headers=headers, timeout=10, verify=False)
        duration = time.time() - start_time
        
        if resp.status_code != 200:
             print(f"  [tiingo] HTTP {resp.status_code} for {ticker}", flush=True)
             _log_api_call('tiingo', duration, False, ticker)
             if resp.status_code == 429:
                  # Use CB for limits? Or just bucket?
                  # Bucket handles known limits. 429 means we messed up or hit daily/monthly.
                  pass
             return pd.DataFrame()

        data = resp.json()
        # Tiingo returns list of dicts: [{'date':..., 'open':...}, ...]
        if not isinstance(data, list) or len(data) == 0:
             _log_api_call('tiingo', duration, False, ticker)
             return pd.DataFrame()

        _CB_TIINGO.record_success()
        _log_api_call('tiingo', duration, True, ticker)
        
        df = pd.DataFrame(data)
        # Columns are usually lowercase already: date, open, high, low, close, volume, adjClose...
        # We need adjClose if available, or close.
        # Let's inspect fields. usually: date, open, high, low, close, volume, adjClose, adjVolume...
        # We prefer adjusted.
        if 'adjClose' in df.columns:
            df['close'] = df['adjClose']
        if 'adjOpen' in df.columns:
            df['open'] = df['adjOpen']
        if 'adjHigh' in df.columns:
            df['high'] = df['adjHigh']
        if 'adjLow' in df.columns:
            df['low'] = df['adjLow']
        if 'adjVolume' in df.columns:
            df['volume'] = df['adjVolume']
            
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        df['ticker'] = ticker
        df['date'] = pd.to_datetime(df['date']).dt.tz_convert(None)
        df = df.sort_values('date').reset_index(drop=True)
        return df

    except Exception as e:
        duration = time.time() - start_time
        _CB_TIINGO.record_failure()
        _log_api_call('tiingo', duration, False, ticker)
        print(f"  [tiingo] Error {ticker}: {str(e)[:50]}", flush=True)
        return pd.DataFrame()

def download_from_twelve_data(ticker: str) -> pd.DataFrame:
    """Fallback downloader using Twelve Data API."""
    if not _CB_TWELVEDATA.allow_request():
        return pd.DataFrame()
        
    if not _LIMITER_TWELVEDATA.consume():
        print(f"  [twelve_data] Rate limited, skipping {ticker}", flush=True)
        return pd.DataFrame()

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import TWELVE_DATA_API_KEY
    
    if not TWELVE_DATA_API_KEY:
        return pd.DataFrame()
    
    url = f"https://api.twelvedata.com/time_series?symbol={ticker}&interval=1day&outputsize=5000&apikey={TWELVE_DATA_API_KEY}&format=csv"
    start_time = time.time()
    
    try:
        # Use requests with retries (via _get_session) to handle SSL/Network flakiness
        session = _get_session("twelve_data")
        resp = session.get(url, timeout=15) # Increased timeout
        duration = time.time() - start_time
        
        if resp.status_code != 200:
             # Log but don't output full error unless debugging
             print(f"  [twelve_data] HTTP {resp.status_code} for {ticker}", flush=True)
             _log_api_call('twelve_data', duration, False, ticker)
             return pd.DataFrame()

        # Check for JSON error inside text (even if 200 OK or CSV endpoint used)
        if '{"code":' in resp.text and '"status":"error"' in resp.text:
             print(f"  [twelve_data] API Error: {resp.text[:100]}", flush=True)
             _log_api_call('twelve_data', duration, False, ticker)
             # If limit reached, trip CB?
             if "limit" in resp.text.lower():
                 _CB_TWELVEDATA.record_failure()
             return pd.DataFrame()

        from io import StringIO
        # Twelve Data CSV responses are semicolon-delimited.
        df = pd.read_csv(StringIO(resp.text), sep=';')
        
        if df.empty or 'datetime' not in df.columns:
            _log_api_call('twelve_data', duration, False, ticker)
            return pd.DataFrame()
            
        _CB_TWELVEDATA.record_success()
        _log_api_call('twelve_data', duration, True, ticker)

        df = df.rename(columns={'datetime': 'date'})
        df.columns = [c.lower() for c in df.columns]
        df['ticker'] = ticker
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        duration = time.time() - start_time
        _CB_TWELVEDATA.record_failure()
        _log_api_call('twelve_data', duration, False, ticker)
        print(f"  [twelve_data] Error {ticker}: {str(e)[:50]}", flush=True)
        return pd.DataFrame()


def download_from_fmp(ticker: str) -> pd.DataFrame:
    """Fallback downloader using Financial Modeling Prep historical endpoint."""
    if not _CB_FMP.allow_request():
        return pd.DataFrame()
    if not _LIMITER_FMP.consume():
        return pd.DataFrame()

    from config import FMP_API_KEY
    if not FMP_API_KEY:
        return pd.DataFrame()

    start_time = time.time()
    try:
        session = _get_session("fmp")
        urls = [
            # Current endpoint family (works on modern plans)
            f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={ticker}&apikey={FMP_API_KEY}",
            # Legacy fallback for older subscribers
            f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={FMP_API_KEY}",
        ]

        for url in urls:
            resp = session.get(url, timeout=12)
            if resp.status_code != 200:
                continue

            payload = resp.json()
            if isinstance(payload, list):
                rows = payload
            elif isinstance(payload, dict):
                rows = payload.get('historical', [])
            else:
                rows = []
            if not rows:
                continue

            df = pd.DataFrame(rows)
            # Some endpoints can provide only close/price; synthesize minimal OHLC if needed.
            if 'price' in df.columns and 'close' not in df.columns:
                df['close'] = df['price']
            if 'close' in df.columns:
                for col in ['open', 'high', 'low']:
                    if col not in df.columns:
                        df[col] = df['close']
            df = _normalize_ohlcv_df(df, ticker)
            if not df.empty:
                duration = time.time() - start_time
                _CB_FMP.record_success()
                _log_api_call('fmp', duration, True, ticker)
                return df

        duration = time.time() - start_time
        _CB_FMP.record_failure()
        _log_api_call('fmp', duration, False, ticker)
        return pd.DataFrame()
    except Exception:
        duration = time.time() - start_time
        _CB_FMP.record_failure()
        _log_api_call('fmp', duration, False, ticker)
        return pd.DataFrame()


def download_from_eodhd(ticker: str) -> pd.DataFrame:
    """Fallback downloader using EODHD end-of-day endpoint."""
    if not _CB_EODHD.allow_request():
        return pd.DataFrame()
    if not _LIMITER_EODHD.consume():
        return pd.DataFrame()

    from config import EODHD_API_KEY
    if not EODHD_API_KEY:
        return pd.DataFrame()

    start = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')
    start_time = time.time()
    try:
        symbols = [ticker, f"{ticker}.US"] if "." not in ticker else [ticker]
        session = _get_session("eodhd")
        for sym in symbols:
            url = (
                f"https://eodhd.com/api/eod/{sym}"
                f"?api_token={EODHD_API_KEY}&fmt=json&period=d&from={start}&to={end}"
            )
            resp = session.get(url, timeout=12)
            if resp.status_code != 200:
                continue
            payload = resp.json()
            if not isinstance(payload, list) or not payload:
                continue
            duration = time.time() - start_time
            df = pd.DataFrame(payload)
            df = _normalize_ohlcv_df(df, ticker)
            ok = not df.empty
            if ok:
                _CB_EODHD.record_success()
            else:
                _CB_EODHD.record_failure()
            _log_api_call('eodhd', duration, ok, ticker)
            return df

        duration = time.time() - start_time
        _CB_EODHD.record_failure()
        _log_api_call('eodhd', duration, False, ticker)
        return pd.DataFrame()
    except Exception:
        duration = time.time() - start_time
        _CB_EODHD.record_failure()
        _log_api_call('eodhd', duration, False, ticker)
        return pd.DataFrame()


def download_batch_from_gemini_cli(tickers: list[str]) -> pd.DataFrame:
    """
    Last-ditch fallback: ask gemini-cli for batched OHLCV bars.
    Uses Flash model first, then default model if rate-limited.
    """
    if not tickers or not _CB_GEMINI.allow_request():
        return pd.DataFrame()

    from config import (
        ENABLE_GEMINI_PRICE_FALLBACK,
        GEMINI_PRICE_PRIMARY_MODEL,
        GEMINI_PRICE_FALLBACK_MODEL,
        GEMINI_PRICE_BATCH_SIZE,
        GEMINI_PRICE_TIMEOUT_SECONDS,
        GEMINI_PRICE_MIN_BARS,
    )
    if not ENABLE_GEMINI_PRICE_FALLBACK:
        return pd.DataFrame()

    recovered_frames = []
    chunk_size = max(1, int(GEMINI_PRICE_BATCH_SIZE or 1))
    valid_tickers = [t.upper() for t in tickers if _is_plausible_ticker(t)]
    if not valid_tickers:
        return pd.DataFrame()

    for idx in range(0, len(valid_tickers), chunk_size):
        chunk = [t.upper() for t in valid_tickers[idx:idx + chunk_size]]
        prompt = (
            "Search the web and return ONLY JSON with this exact schema:\n"
            "{\"tickers\":[{\"ticker\":\"AAPL\",\"bars\":[{\"date\":\"YYYY-MM-DD\",\"open\":0,\"high\":0,\"low\":0,\"close\":0,\"volume\":0}]}]}\n"
            "Rules: include approximately the most recent 260 daily bars per ticker; no markdown; no commentary.\n"
            f"Tickers: {', '.join(chunk)}"
        )

        raw, err = _run_gemini_cli_prompt(
            prompt,
            model=GEMINI_PRICE_PRIMARY_MODEL,
            timeout=int(GEMINI_PRICE_TIMEOUT_SECONDS or 180),
        )
        if err in ("rate_limit", "quota") and GEMINI_PRICE_FALLBACK_MODEL is not None:
            fallback_model = (GEMINI_PRICE_FALLBACK_MODEL or None)
            raw, err = _run_gemini_cli_prompt(
                prompt,
                model=fallback_model,
                timeout=int(GEMINI_PRICE_TIMEOUT_SECONDS or 180),
            )

        if err:
            _CB_GEMINI.record_failure()
            continue

        parsed = _parse_gemini_price_payload(raw)
        if not parsed:
            _CB_GEMINI.record_failure()
            continue

        chunk_ok = False
        for ticker, bars in parsed.items():
            if ticker not in chunk:
                continue
            df = _normalize_ohlcv_df(pd.DataFrame(bars), ticker)
            if len(df) < int(GEMINI_PRICE_MIN_BARS or 200):
                continue
            recovered_frames.append(df)
            chunk_ok = True

        if chunk_ok:
            _CB_GEMINI.record_success()
        else:
            _CB_GEMINI.record_failure()

    if recovered_frames:
        return pd.concat(recovered_frames, ignore_index=True)
    return pd.DataFrame()


def download_batch_with_fallback(tickers: list) -> pd.DataFrame:
    """
    Downloads a batch of tickers with automatic fallback and smart API rotation.
    On rate limit: immediately moves to next API (no retries), marks API for 60s cooldown.
    Chain: yfinance -> alpaca -> tiingo -> stooq -> twelve -> finnhub -> polygon
           -> alpha vantage -> fmp -> eodhd -> gemini-cli (last-ditch)
    """
    from concurrent.futures import ThreadPoolExecutor
    
    # 1. Try yfinance batch (Fastest) - if available
    batch_df = pd.DataFrame()
    batch_tickers = set()
    
    if _CB_YFINANCE.allow_request():
        print(f"  Attempting yfinance batch for {len(tickers)} tickers...")
        try:
            batch_df = download_batch(tickers)
            if not batch_df.empty:
                batch_tickers = set(batch_df['ticker'].unique())
                success_rate = len(batch_tickers) / len(tickers)
                print(f"  yfinance batch: {len(batch_tickers)}/{len(tickers)} ({success_rate:.0%}) succeeded")
                _CB_YFINANCE.record_success()
                
                # Add backoff if batch had low success rate (rate limiting detected)
                if success_rate < 0.5:
                    print("  Rate limit detected, backing off 30s...")
                    time.sleep(30)
                
                # If all tickers succeeded, return immediately 
                if len(batch_tickers) >= len(tickers):
                    return batch_df
            else:
                _CB_YFINANCE.record_failure()
                 
        except Exception as e:
            _CB_YFINANCE.record_failure()
            print(f"  yfinance batch failed: {e}")
    
    # 2. Identify missing tickers and use parallel fallback for those only
    missing_tickers = [t for t in tickers if t not in batch_tickers]
    
    if missing_tickers:
        print(f"  {len(missing_tickers)} tickers missing. Attempting individual fallback...")
    
    all_dfs = [batch_df] if not batch_df.empty else []
    recovered_tickers = batch_tickers.copy()
    
    missing_tickers = [t for t in tickers if t not in batch_tickers]
    print(f"  [debug] Batch size: {len(tickers)} | Success: {len(batch_tickers)} | Missing: {len(missing_tickers)}", flush=True)

    if missing_tickers:
        print(f"  [debug] Starting fallbacks for: {missing_tickers[:5]}...", flush=True)

    def fetch_single_smart(ticker):
        """Try APIs in order, respecting rate limits."""
        # Add staggered delay to prevent parallel API exhaustion
        time.sleep(random.uniform(0.1, 0.5))
        
        # yfinance single
        if _CB_YFINANCE.allow_request():
            try:
                res = split_and_download_ticker(ticker)
                if not res.empty:
                    _CB_YFINANCE.record_success()
                    return res
            except Exception as e:
                _CB_YFINANCE.record_failure()
        
        # Alpaca (High Speed: 200/min)
        print(f"  [debug] Trying Alpaca for {ticker}", flush=True)
        res = download_from_alpaca(ticker)
        if not res.empty:
            return res

        # Tiingo (Medium Speed: 500/hr)
        res = download_from_tiingo(ticker)
        if not res.empty:
            return res

        # Stooq (checks CB internally)
        res = download_from_stooq(ticker)
        if not res.empty:
            return res
            
        # Twelve Data (checks CB & TokenBucket)
        res = download_from_twelve_data(ticker)
        if not res.empty:
            return res
        
        # Finnhub (checks CB & TokenBucket internally)
        res = download_from_finnhub(ticker)
        if not res.empty:
            return res
        
        # Polygon (checks CB & TokenBucket internally)
        res = download_from_polygon(ticker)
        if not res.empty:
            return res

        # Financial Modeling Prep
        res = download_from_fmp(ticker)
        if not res.empty:
            return res

        # EODHD
        res = download_from_eodhd(ticker)
        if not res.empty:
            return res
        
        return pd.DataFrame()  # All APIs failed or rate limited

    # Run parallel with defined workers (or 1 for safety)
    try:
        from config import SCANNER_WORKERS
        workers = SCANNER_WORKERS
    except ImportError:
        workers = 1

    # Only run parallel fallback if there are missing tickers
    if missing_tickers:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(fetch_single_smart, missing_tickers))
            for r in results:
                if not r.empty:
                    all_dfs.append(r)
                    recovered_tickers.add(r['ticker'].iloc[0])
            
    # 3. Alpha Vantage Fallback for remaining gaps (sequential, slower)
    missing = [t for t in tickers if t not in recovered_tickers]
    
    # We only try AV if NOT circuit broken. 
    # (Token check is inside download_from_alpha_vantage)
    if missing and _CB_ALPHAVANTAGE.allow_request():
        print(f"  {len(missing)} tickers still missing. Attempting Alpha Vantage...")
        for ticker in missing:
            # download_from_alpha_vantage handles token bucket wait/skip
            res = download_from_alpha_vantage(ticker)
            if not res.empty: 
                all_dfs.append(res)
                recovered_tickers.add(ticker)
                print(f"    - {ticker} recovered via Alpha Vantage")
            else:
                 # If it failed or was skipped due to rate limit, we move closely to next?
                 # If rate limited, download_from_alpha_vantage returns empty immediate.
                 # Stop trying if circuit breaker tripped inside or just keep going?
                 # But we don't want to infinite loop if no one succeeds.
                 # Just break loop if CB tripped?
                 if _CB_ALPHAVANTAGE.state == "OPEN":
                     break

    # 4. FMP fallback for any remaining gaps
    missing = [t for t in tickers if t not in recovered_tickers]
    if missing and _CB_FMP.allow_request():
        print(f"  {len(missing)} tickers still missing. Attempting FMP...")
        for ticker in missing:
            res = download_from_fmp(ticker)
            if not res.empty:
                all_dfs.append(res)
                recovered_tickers.add(ticker)

    # 5. EODHD fallback for remaining gaps
    missing = [t for t in tickers if t not in recovered_tickers]
    if missing and _CB_EODHD.allow_request():
        print(f"  {len(missing)} tickers still missing. Attempting EODHD...")
        for ticker in missing:
            res = download_from_eodhd(ticker)
            if not res.empty:
                all_dfs.append(res)
                recovered_tickers.add(ticker)

    # 6. Gemini CLI fallback (batched) as absolute last resort
    missing = [t for t in tickers if t not in recovered_tickers]
    plausible_missing = [t for t in missing if _is_plausible_ticker(t)]
    if plausible_missing and _CB_GEMINI.allow_request():
        print(f"  {len(plausible_missing)} tickers still missing. Attempting gemini-cli fallback...")
        gem_df = download_batch_from_gemini_cli(plausible_missing)
        if not gem_df.empty:
            all_dfs.append(gem_df)
            
    return pd.concat(all_dfs) if all_dfs else pd.DataFrame()

def download_with_fallback(ticker: str) -> pd.DataFrame:
    """Single-ticker fallback chain (same order as batch, ending with gemini-cli)."""
    # Try yfinance first
    if _CB_YFINANCE.allow_request():
        try:
             df = split_and_download_ticker(ticker)
             if not df.empty:
                 _CB_YFINANCE.record_success()
                 return df
        except Exception:
             _CB_YFINANCE.record_failure()
    
    # Try Stooq (checks CB internally)
    df = download_from_stooq(ticker)
    if not df.empty:
        return df
        
    # Try Twelve Data
    df = download_from_twelve_data(ticker)
    if not df.empty:
        return df
    
    # Try Finnhub (checks CB/TB internally)
    df = download_from_finnhub(ticker)
    if not df.empty:
        return df
    
    # Try Polygon (checks CB/TB internally)
    df = download_from_polygon(ticker)
    if not df.empty:
        return df
    
    # Try Alpha Vantage (checks CB/TB internally)
    # Note: download_from_alpha_vantage has its own token consumer.
    df = download_from_alpha_vantage(ticker)
    if not df.empty:
        return df

    # Try FMP
    df = download_from_fmp(ticker)
    if not df.empty:
        return df

    # Try EODHD
    df = download_from_eodhd(ticker)
    if not df.empty:
        return df

    # Try gemini-cli batch fallback for this one ticker
    df = download_batch_from_gemini_cli([ticker])
    if not df.empty:
        return df
    
    
    return pd.DataFrame()

# =============================================================================
# FUNDAMENTALS & YIELD FETCHING
# =============================================================================

def _fetch_fundamentals_polygon(ticker: str) -> dict:
    """Fetch fundamentals from Polygon.io Ticker Details v3."""
    if not _CB_POLYGON.allow_request(): return None
    if not _LIMITER_POLYGON.consume(): return None

    from config import POLYGON_API_KEY
    if not POLYGON_API_KEY: return None

    try:
        url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={POLYGON_API_KEY}"
        session = _get_session("polygon")
        resp = session.get(url, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'OK' and 'results' in data:
                res = data['results']
                # Polygon v3 Ticker Details doesn't always have yield directly, 
                # often requires 'Snapshot' or 'Dividends' endpoint.
                # Checking 'market_cap' usually works.
                return {
                    'market_cap': res.get('market_cap'),
                    'source': 'polygon',
                    # Polygon details doesn't have yield easily without extra calls.
                    # We will return what we can.
                    'currency': res.get('currency_name')
                }
    except Exception:
        _CB_POLYGON.record_failure()
    return None

def _fetch_fundamentals_alphavantage(ticker: str) -> dict:
    """Fetch fundamentals from Alpha Vantage OVERVIEW."""
    if not _CB_ALPHAVANTAGE.allow_request(): return None
    if not _ALPHA_VANTAGE_KEYS:
        return None

    try:
        session = _get_session("alpha_vantage")
        for _ in range(len(_ALPHA_VANTAGE_KEYS)):
            if not _LIMITER_ALPHAVANTAGE.consume():
                return None

            api_key = _next_alpha_vantage_key()
            if not api_key:
                return None

            url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}"
            resp = session.get(url, timeout=10)
            if resp.status_code != 200:
                continue

            data = resp.json()
            if "Symbol" not in data:
                continue

            # AV returns numeric fields as strings
            try:
                div_yield = float(data.get("DividendYield", 0))
            except Exception:
                div_yield = 0.0

            try:
                pe = float(data.get("PERatio", 0))
            except Exception:
                pe = 0.0

            try:
                mc = float(data.get("MarketCapitalization", 0))
            except Exception:
                mc = 0.0

            return {
                'dividend_yield': div_yield,
                'pe_ratio': pe,
                'market_cap': mc,
                'sector': data.get('Sector'),
                'industry': data.get('Industry'),
                'source': 'alpha_vantage'
            }
    except Exception:
        pass

    _CB_ALPHAVANTAGE.record_failure()
    return None

def _fetch_fundamentals_finnhub(ticker: str) -> dict:
    """Fetch fundamentals from Finnhub Basic Financials."""
    if not _CB_FINNHUB.allow_request(): return None
    if not _LIMITER_FINNHUB.consume(): return None

    from config import FINNHUB_API_KEY
    if not FINNHUB_API_KEY: return None

    try:
        url = f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={FINNHUB_API_KEY}"
        session = _get_session("finnhub")
        resp = session.get(url, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            if "metric" in data:
                m = data["metric"]
                # dividendYieldIndicatedAnnual is usually good
                dy = m.get("dividendYieldIndicatedAnnual", 0) or m.get("dividendYield5Y", 0)
                # Finnhub returns percentage (e.g. 1.5 for 1.5%), we need decimal usually or check
                # User wants 4.2% -> 0.042? 
                # YFinance returns 0.042. AV returns 0.042. 
                # Finnhub documentation says "Annual Dividend Yield". usually %
                if dy and dy > 0.5: # massive heuristic: if > 0.5, likely %.
                     dy = dy / 100.0
                     
                return {
                    'dividend_yield': dy,
                    'market_cap': m.get("marketCapitalization", 0),
                    'pe_ratio': m.get("peBasicExclExtraTTM", 0),
                    'source': 'finnhub'
                }
    except Exception:
        _CB_FINNHUB.record_failure()
    return None

def get_ticker_fundamentals(ticker: str) -> dict:
    """
    Robustly fetch fundamentals (Yield, PE, Market Cap).
    Cycles through configured APIs: Finnhub -> Alpha Vantage -> Polygon -> YFinance.
    """
    
    # 1. Finnhub (Good metrics, fast)
    res = _fetch_fundamentals_finnhub(ticker)
    if res and res.get('dividend_yield', 0) > 0:
        return res
        
    # 2. Alpha Vantage (Solid overview, but slow/rate limited)
    res = _fetch_fundamentals_alphavantage(ticker)
    if res and res.get('dividend_yield', 0) > 0:
        return res
        
    # 3. YFinance (Fallback, has info but relies on yahoo)
    try:
        if _CB_YFINANCE.allow_request():
            info = yf.Ticker(ticker).info
            dy = info.get('dividendYield', 0) or 0
            if dy > 0:
                _CB_YFINANCE.record_success()
                return {
                    'dividend_yield': dy,
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'sector': info.get('sector'),
                    'industry': info.get('industry'),
                    'source': 'yfinance'
                }
    except:
        _CB_YFINANCE.record_failure()
        
    # 4. Default / Empty
    return {
        'dividend_yield': 0.0,
        'pe_ratio': 0.0,
        'market_cap': 0.0,
        'source': 'none'
    }

def estimate_dividend_yield_from_history(ticker: str, trailing_days: int, max_age_days: int) -> float:
    """
    Estimate dividend yield from recent dividend history.
    Uses trailing dividend sum if the most recent dividend is not stale.
    """
    try:
        import yfinance as yf
        from datetime import datetime, timedelta

        stock = yf.Ticker(ticker)
        dividends = stock.dividends
        if dividends is None or dividends.empty:
            return 0.0

        last_date = dividends.index.max()
        if not last_date:
            return 0.0

        if (datetime.now() - last_date.to_pydatetime()).days > max_age_days:
            return 0.0

        cutoff = datetime.now() - timedelta(days=trailing_days)
        trailing_divs = dividends[dividends.index >= cutoff]
        trailing_sum = float(trailing_divs.sum()) if not trailing_divs.empty else 0.0
        if trailing_sum <= 0:
            return 0.0

        price = 0.0
        try:
            info = stock.info or {}
            price = info.get('regularMarketPrice') or info.get('currentPrice') or 0.0
        except Exception:
            price = 0.0

        if not price:
            hist = stock.history(period="5d")
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])

        if price <= 0:
            return 0.0

        return trailing_sum / price
    except Exception:
        return 0.0


