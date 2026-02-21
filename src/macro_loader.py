"""
Macro Data Loader
==================
Fetches and caches macroeconomic indicators from FRED.
Designed to minimize API calls with aggressive caching.
"""
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional

# Get project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_SCRIPT_DIR) == 'src':
    PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
else:
    PROJECT_ROOT = _SCRIPT_DIR

CACHE_DIR = os.path.join(PROJECT_ROOT, 'data', 'macro_cache')

# Import config
import sys
sys.path.insert(0, PROJECT_ROOT)
from config import FRED_API_KEY, FRED_SERIES, MACRO_CACHE_HOURS, BENCHMARK_BLEND

# Cache duration by frequency (in hours)
CACHE_DURATION = {
    'daily': 18,     # Refresh daily data after 18 hours (next trading day)
    'weekly': 168,   # Refresh weekly data after 7 days
    'monthly': 720,  # Refresh monthly data after 30 days
}


def _get_cache_path(series_id: str) -> str:
    """Get cache file path for a FRED series."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return os.path.join(CACHE_DIR, f'{series_id}.json')


def _get_cache_hours(series_id: str) -> int:
    """Get cache duration based on series frequency."""
    freq = FRED_SERIES.get(series_id, 'daily')
    return CACHE_DURATION.get(freq, MACRO_CACHE_HOURS)


def _is_cache_valid(cache_path: str, series_id: str = None) -> bool:
    """
    Check if cache file exists and is fresh enough based on series frequency.
    Daily series: cache 18h, Weekly: cache 7 days, Monthly: cache 30 days.
    """
    if not os.path.exists(cache_path):
        return False
    
    # Determine cache duration based on series frequency
    if series_id:
        max_age_hours = _get_cache_hours(series_id)
    else:
        max_age_hours = MACRO_CACHE_HOURS
    
    mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
    age = datetime.now() - mtime
    return age < timedelta(hours=max_age_hours)


def _load_from_cache(cache_path: str) -> Optional[pd.Series]:
    """Load series from cache file."""
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
        series = pd.Series(data['values'], name=data['series_id'])
        series.index = pd.to_datetime(data['dates'])
        return series
    except Exception as e:
        print(f"Cache load error: {e}")
        return None


def _save_to_cache(series: pd.Series, series_id: str, cache_path: str):
    """Save series to cache file."""
    try:
        data = {
            'series_id': series_id,
            'last_updated': datetime.now().isoformat(),
            'dates': [d.isoformat() for d in series.index],
            'values': series.tolist()
        }
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Cache save error: {e}")


def fetch_fred_series(series_id: str, force_refresh: bool = False) -> Optional[pd.Series]:
    """
    Fetch a single FRED series with caching.
    
    Args:
        series_id: FRED series identifier (e.g., 'DGS10')
        force_refresh: Bypass cache and fetch fresh data
        
    Returns:
        pandas Series with datetime index, or None if error
    """
    cache_path = _get_cache_path(series_id)
    
    # Try cache first (unless forcing refresh)
    if not force_refresh and _is_cache_valid(cache_path, series_id):
        cached = _load_from_cache(cache_path)
        if cached is not None:
            print(f"✓ {series_id}: Loaded from cache")
            return cached
    
    # Fetch from FRED API
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        
        # Fetch last 2 years of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        series = fred.get_series(series_id, start_date, end_date)
        series = series.dropna()
        
        if not series.empty:
            _save_to_cache(series, series_id, cache_path)
            print(f"✓ {series_id}: Fetched from FRED ({len(series)} points)")
            return series
        else:
            print(f"✗ {series_id}: No data returned")
            return None
            
    except ImportError:
        print("fredapi not installed. Run: pip install fredapi")
        return None
    except Exception as e:
        print(f"✗ {series_id}: FRED API error - {e}")
        # Try to return stale cache if available
        if os.path.exists(cache_path):
            return _load_from_cache(cache_path)
        return None


def fetch_all_macro_data(force_refresh: bool = False) -> Dict[str, pd.Series]:
    """
    Fetch all configured FRED series.
    
    Returns:
        Dict mapping series_id -> pandas Series
    """
    print(f"\n📊 Loading Macro Data (cache age: {MACRO_CACHE_HOURS}h)...")
    
    data = {}
    for series_id in FRED_SERIES.keys():
        series = fetch_fred_series(series_id, force_refresh)
        if series is not None:
            data[series_id] = series
    
    print(f"Loaded {len(data)}/{len(FRED_SERIES)} macro series\n")
    return data


def get_macro_features() -> Dict[str, float]:
    """
    Get latest macro indicators as features for ML model.
    
    Returns:
        Dict of feature_name -> value (most recent observation)
    """
    data = fetch_all_macro_data()
    features = {}
    
    for series_id, series in data.items():
        if len(series) > 0:
            # Latest value
            features[f'macro_{series_id}'] = float(series.iloc[-1])
            
            # 1-week change (if enough data)
            if len(series) >= 5:
                features[f'macro_{series_id}_chg1w'] = float(series.iloc[-1] - series.iloc[-5])
            
            # 1-month change (if enough data)
            if len(series) >= 22:
                features[f'macro_{series_id}_chg1m'] = float(series.iloc[-1] - series.iloc[-22])
    
    return features


def get_macro_features_asof(asof_date, data: Optional[Dict[str, pd.Series]] = None) -> Dict[str, float]:
    """
    Get macro indicators using only observations at or before ``asof_date``.

    Note:
        This is date-based point-in-time filtering. It does not model publication
        lag revisions beyond each series timestamp.
    """
    asof_ts = pd.to_datetime(asof_date, errors='coerce')
    if pd.isna(asof_ts):
        return {}
    asof_ts = pd.Timestamp(asof_ts).normalize()

    macro_data = data if data is not None else fetch_all_macro_data()
    features: Dict[str, float] = {}

    for series_id, series in macro_data.items():
        if series is None or len(series) == 0:
            continue
        try:
            clean = series.dropna().sort_index()
            idx = pd.to_datetime(clean.index, errors='coerce')
            valid = ~idx.isna()
            if not bool(valid.any()):
                continue
            idx = pd.DatetimeIndex(idx[valid])
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            clean = pd.Series(clean.iloc[valid].to_numpy(), index=idx)
            asof_slice = clean[clean.index <= asof_ts]
        except Exception:
            continue
        if asof_slice.empty:
            continue

        features[f'macro_{series_id}'] = float(asof_slice.iloc[-1])
        if len(asof_slice) >= 5:
            features[f'macro_{series_id}_chg1w'] = float(asof_slice.iloc[-1] - asof_slice.iloc[-5])
        if len(asof_slice) >= 22:
            features[f'macro_{series_id}_chg1m'] = float(asof_slice.iloc[-1] - asof_slice.iloc[-22])

    return features


def get_macro_regime() -> str:
    """
    Classify current macro environment using multiple indicators.
    
    Returns:
        'RISK_ON', 'RISK_OFF', or 'NEUTRAL'
    """
    data = fetch_all_macro_data()
    
    risk_score = 0
    signals = []
    
    # 1. VIX Level (fear gauge)
    if 'VIXCLS' in data and len(data['VIXCLS']) > 0:
        vix = data['VIXCLS'].iloc[-1]
        if vix > 30:
            risk_score -= 2
            signals.append(f"VIX high ({vix:.1f})")
        elif vix > 20:
            risk_score -= 1
            signals.append(f"VIX elevated ({vix:.1f})")
        elif vix < 15:
            risk_score += 1
            signals.append(f"VIX low ({vix:.1f})")
    
    # 2. Yield curve: inverted = bearish
    if 'T10Y2Y' in data and len(data['T10Y2Y']) > 0:
        spread = data['T10Y2Y'].iloc[-1]
        if spread < -0.5:
            risk_score -= 2
            signals.append(f"Yield curve deeply inverted ({spread:.2f})")
        elif spread < 0:
            risk_score -= 1
            signals.append(f"Yield curve inverted ({spread:.2f})")
        elif spread > 1.0:
            risk_score += 1
            signals.append(f"Yield curve normal ({spread:.2f})")
    
    # 3. High yield spread: level and change
    if 'BAMLH0A0HYM2' in data and len(data['BAMLH0A0HYM2']) >= 22:
        hy_spread = data['BAMLH0A0HYM2']
        hy_level = hy_spread.iloc[-1]
        hy_change = hy_spread.iloc[-1] - hy_spread.iloc[-22]
        
        if hy_level > 5:
            risk_score -= 2
            signals.append(f"HY spread wide ({hy_level:.1f}%)")
        elif hy_change > 0.5:
            risk_score -= 1
            signals.append(f"HY spread widening (+{hy_change:.2f})")
        elif hy_change < -0.5:
            risk_score += 1
            signals.append(f"HY spread tightening ({hy_change:.2f})")
    
    # 4. Financial Stress Index
    if 'STLFSI4' in data and len(data['STLFSI4']) > 0:
        stress = data['STLFSI4'].iloc[-1]
        if stress > 1.0:
            risk_score -= 2
            signals.append(f"Financial stress high ({stress:.2f})")
        elif stress > 0:
            risk_score -= 1
            signals.append(f"Financial stress elevated ({stress:.2f})")
        elif stress < -0.5:
            risk_score += 1
            signals.append(f"Financial stress low ({stress:.2f})")
    
    # 5. Jobless claims: rising = bearish
    if 'ICSA' in data and len(data['ICSA']) >= 4:
        claims = data['ICSA']
        claims_change = (claims.iloc[-1] - claims.iloc[-4]) / claims.iloc[-4]
        if claims_change > 0.15:
            risk_score -= 1
            signals.append(f"Claims rising ({claims_change:.1%})")
        elif claims_change < -0.1:
            risk_score += 1
            signals.append(f"Claims falling ({claims_change:.1%})")
    
    # 6. Fed Balance Sheet trend (QE = bullish, QT = bearish)
    if 'WALCL' in data and len(data['WALCL']) >= 12:
        fed_assets = data['WALCL']
        fed_change = (fed_assets.iloc[-1] - fed_assets.iloc[-12]) / fed_assets.iloc[-12]
        if fed_change > 0.02:
            risk_score += 1
            signals.append(f"Fed expanding ({fed_change:.1%})")
        elif fed_change < -0.02:
            risk_score -= 1
            signals.append(f"Fed contracting ({fed_change:.1%})")
    
    # 7. Consumer sentiment
    if 'UMCSENT' in data and len(data['UMCSENT']) >= 2:
        sentiment = data['UMCSENT']
        if sentiment.iloc[-1] > sentiment.iloc[-2]:
            risk_score += 1
        else:
            risk_score -= 1
    
    # Classify regime
    if risk_score >= 3:
        return 'RISK_ON'
    elif risk_score <= -3:
        return 'RISK_OFF'
    else:
        return 'NEUTRAL'


def fetch_benchmark_returns(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Fetch returns for benchmark blend components.
    
    Returns:
        DataFrame with daily returns for each benchmark component and blended return
    """
    import yfinance as yf
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching benchmark data: {list(BENCHMARK_BLEND.keys())}")
    
    tickers = list(BENCHMARK_BLEND.keys())
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    
    # Calculate daily returns
    returns = data.pct_change().dropna()
    
    # Calculate blended return
    blended = pd.Series(0.0, index=returns.index)
    for ticker, weight in BENCHMARK_BLEND.items():
        if ticker in returns.columns:
            blended += returns[ticker] * weight
    
    returns['BENCHMARK_BLEND'] = blended
    
    return returns


if __name__ == "__main__":
    # Test the loader
    print("Testing Macro Data Loader...")
    
    # Fetch all data
    data = fetch_all_macro_data()
    
    # Get features
    features = get_macro_features()
    print(f"\nMacro Features ({len(features)} total):")
    for k, v in list(features.items())[:10]:
        print(f"  {k}: {v:.4f}")
    
    # Get regime
    regime = get_macro_regime()
    print(f"\nCurrent Macro Regime: {regime}")
    
    # Test benchmark
    bench = fetch_benchmark_returns()
    print(f"\nBenchmark Returns Shape: {bench.shape}")
    print(f"Cumulative Benchmark Return (YTD): {(1 + bench['BENCHMARK_BLEND']).prod() - 1:.2%}")
