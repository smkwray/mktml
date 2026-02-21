"""
Sector and Country Analysis Module
===================================
Provides metadata enrichment and aggregation for sector/country based analysis.
"""
import json
import os
from datetime import datetime, timedelta
from typing import Optional

# Cache file for ticker metadata
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_CACHE_FILE = os.path.join(PROJECT_ROOT, "data", "ticker_metadata.json")

def _load_cache() -> dict:
    """Load cached ticker metadata."""
    if os.path.exists(METADATA_CACHE_FILE):
        try:
            with open(METADATA_CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def _save_cache(cache: dict):
    """Save ticker metadata to cache."""
    os.makedirs(os.path.dirname(METADATA_CACHE_FILE), exist_ok=True)
    with open(METADATA_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

# Circuit breaker state for yfinance
_YFINANCE_BLOCKED_UNTIL = 0

def get_ticker_metadata(ticker: str, fetch_online: bool = True) -> dict:
    """
    Fetches sector, industry, and country for a ticker.
    Args:
        fetch_online: If False, only check cache and static universe.
                      If True, try APIs if cache missing.
    Strategy:
    1. Check local DB cache (permanent storage)
    2. Check Universe lists (for ETFs)
    3. Fetch from API (Tiingo -> Finnhub -> Polygon -> AlphaVantage -> yfinance(last resort)) -> Store in DB
    """
    from src.storage import get_metadata, save_metadata
    from src.universe import get_etf_universe
    import time
    
    # 1. Check DB Cache
    cached = get_metadata(ticker)
    if cached and cached.get('sector') not in [None, 'Unknown']:
        return cached

    meta = {
        'ticker': ticker,
        'sector': 'Unknown', 
        'industry': 'Unknown', 
        'country': 'Unknown', 
        'market_cap': 0
    }

    # 2. Check Static Universe (Fastest for ETFs)
    etfs = get_etf_universe()
    found_etf = False
    for category, tickers in etfs.items():
        if ticker in tickers:
            meta['sector'] = 'ETF'
            meta['industry'] = category
            meta['country'] = 'USA' # Default for most
            found_etf = True
            break
            
    if found_etf:
        save_metadata(meta)
        return meta
        
    # If online fetching is disabled, stop here
    if not fetch_online:
        return meta

    # 3. Tiingo (First choice if Key available)
    try:
        tiingo_meta = get_tiingo_metadata(ticker)
        if tiingo_meta.get('sector', 'Unknown') != 'Unknown':
            save_metadata(tiingo_meta)
            return tiingo_meta
    except Exception:
        pass

    # 4. Finnhub (Good fallback, 60 calls/min)
    try:
        fh_meta = get_finnhub_metadata(ticker)
        if fh_meta.get('sector', 'Unknown') != 'Unknown':
            save_metadata(fh_meta)
            return fh_meta
    except Exception:
        pass

    # 5. Twelve Data (Reference, 8 calls/min)
    try:
        td_meta = get_twelve_data_metadata(ticker)
        if td_meta.get('sector', 'Unknown') != 'Unknown':
            save_metadata(td_meta)
            return td_meta
    except Exception:
        pass

    # 6. Polygon.io (Reference API, 5 calls/min free)
    try:
        poly_meta = get_polygon_metadata(ticker)
        if poly_meta.get('sector', 'Unknown') != 'Unknown':
            save_metadata(poly_meta)
            return poly_meta
    except Exception:
        pass
        
    # 7. Alpha Vantage (Overview API, 5 calls/min)
    try:
        av_meta = get_alphavantage_metadata(ticker)
        if av_meta.get('sector', 'Unknown') != 'Unknown':
            save_metadata(av_meta)
            return av_meta
    except Exception:
        pass

    # 8. Fetch from yfinance (Last Resort)
    # Check circuit breaker
    global _YFINANCE_BLOCKED_UNTIL
    if time.time() < _YFINANCE_BLOCKED_UNTIL:
        # Silently skip if blocked
        return meta

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if info and info.get('regularMarketPrice') is not None:
             meta['sector'] = info.get('sector', 'Unknown')
             meta['industry'] = info.get('industry', 'Unknown')
             meta['country'] = info.get('country', 'Unknown')
             meta['market_cap'] = info.get('marketCap', 0)
             
             # If good data, save
             if meta['sector'] != 'Unknown':
                 save_metadata(meta)
                 return meta
        else:
             pass

    except Exception as e:
        err_str = str(e).lower()
        if "too many requests" in err_str or "rate limit" in err_str or "429" in err_str:
            print(f"  [sector_country] yfinance rate limited on {ticker}. Pausing yfinance calls for 60s.")
            _YFINANCE_BLOCKED_UNTIL = time.time() + 60
        else:
             print(f"  [sector_country] Error for {ticker}: {e}", flush=True)

    # 8. Final Inference/Cleanup
    if meta['sector'] == 'Unknown':
        if ticker.endswith('X') or ticker in ['SPY', 'QQQ', 'VTI', 'VOO']:
            meta['sector'] = 'ETF'
        elif ticker in ['EIS', 'ISRA']:
             meta['country'] = 'Israel'
             meta['sector'] = 'ETF'

    return meta

def get_tiingo_metadata(ticker: str) -> dict:
    """Fetches metadata from Tiingo API."""
    import requests
    from config import TIINGO_API_KEY
    
    if not TIINGO_API_KEY:
        return {'sector': 'Unknown'}
        
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token {TIINGO_API_KEY}'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=3) # Fast timeout
        if response.status_code == 200:
             # Tiingo daily still generic, but checking anyway
             pass
    except Exception:
        pass
        
    return {'sector': 'Unknown'}

def get_finnhub_metadata(ticker: str) -> dict:
    """Fetches company profile from Finnhub."""
    import requests
    from config import FINNHUB_API_KEY
    
    if not FINNHUB_API_KEY:
        return {'sector': 'Unknown'}
        
    url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={FINNHUB_API_KEY}"
    
    try:
        response = requests.get(url, timeout=3) # Fast timeout
        if response.status_code == 200:
            data = response.json()
            if data and 'finnhubIndustry' in data:
                return {
                    'ticker': ticker,
                    'sector': data.get('finnhubIndustry', 'Unknown'),
                    'industry': data.get('finnhubIndustry', 'Unknown'),
                    'country': data.get('country', 'Unknown'),
                    'market_cap': data.get('marketCapitalization', 0) * 1e6
                }
    except Exception:
        pass
        
    return {'sector': 'Unknown'}

def get_polygon_metadata(ticker: str) -> dict:
    """Fetches metadata from Polygon.io Ticker Details v3."""
    import requests
    from config import POLYGON_API_KEY
    
    if not POLYGON_API_KEY:
        return {'sector': 'Unknown'}
        
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={POLYGON_API_KEY}"
    
    try:
        response = requests.get(url, timeout=3) # Fast timeout
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'OK' and 'results' in data:
                res = data['results']
                return {
                    'ticker': ticker,
                    'sector': res.get('sic_description', 'Unknown'), # Best approx usually
                    'industry': res.get('sic_description', 'Unknown'),
                    'country': res.get('locale', 'Unknown').upper(),
                    'market_cap': res.get('market_cap', 0)
                }
    except Exception:
        pass
    
    return {'sector': 'Unknown'}

def get_alphavantage_metadata(ticker: str) -> dict:
    """Fetches metadata from Alpha Vantage OVERVIEW."""
    import requests
    from config import ALPHA_VANTAGE_API_KEY, ALPHA_VANTAGE_API_KEYS

    keys = [k for k in (ALPHA_VANTAGE_API_KEYS or []) if k]
    api_key = keys[0] if keys else ALPHA_VANTAGE_API_KEY
    if not api_key:
        return {'sector': 'Unknown'}
        
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}"
    
    try:
        response = requests.get(url, timeout=3) # Fast timeout
        if response.status_code == 200:
            data = response.json()
            # AV returns JSON with keys: Sector, Industry, Country, MarketCapitalization
            if 'Sector' in data:
                 return {
                    'ticker': ticker,
                    'sector': data.get('Sector', 'Unknown'),
                    'industry': data.get('Industry', 'Unknown'),
                    'country': data.get('Country', 'Unknown'),
                    'market_cap': float(data.get('MarketCapitalization', 0))
                }
    except Exception:
        pass
        
    return {'sector': 'Unknown'}



def aggregate_by_sector(results: list) -> dict:
    """
    Aggregates scan results by sector.
    
    Returns:
        {
            'Technology': {'buy': 5, 'sell': 2, 'hold': 10, 'avg_conf': 0.52},
            ...
        }
    """
    from collections import defaultdict
    
    sector_stats = defaultdict(lambda: {'buy': 0, 'sell': 0, 'hold': 0, 'confidences': []})
    
    for r in results:
        ticker = r.get('ticker', '')
        # Only check cache/static. Do NOT fetch online to save API calls for ML.
        meta = get_ticker_metadata(ticker, fetch_online=False) 
        sector = meta.get('sector', 'Unknown')
        
        signal = r.get('signal_type', 'HOLD').upper()
        if signal == 'BUY':
            sector_stats[sector]['buy'] += 1
        elif signal == 'SELL':
            sector_stats[sector]['sell'] += 1
        else:
            sector_stats[sector]['hold'] += 1
        
        sector_stats[sector]['confidences'].append(r.get('confidence', 0.5))
    
    # Compute averages
    output = {}
    for sector, stats in sector_stats.items():
        confs = stats['confidences']
        output[sector] = {
            'buy': stats['buy'],
            'sell': stats['sell'],
            'hold': stats['hold'],
            'total': stats['buy'] + stats['sell'] + stats['hold'],
            'buy_pct': stats['buy'] / max(1, stats['buy'] + stats['sell'] + stats['hold']),
            'avg_conf': sum(confs) / len(confs) if confs else 0.5
        }
    
    return dict(sorted(output.items(), key=lambda x: x[1]['buy_pct'], reverse=True))


def aggregate_by_country(results: list) -> dict:
    """
    Aggregates scan results by country.
    """
    from collections import defaultdict
    
    country_stats = defaultdict(lambda: {'buy': 0, 'sell': 0, 'hold': 0, 'confidences': []})
    
    for r in results:
        ticker = r.get('ticker', '')
        # Only check cache/static. Do NOT fetch online to save API calls for ML.
        meta = get_ticker_metadata(ticker, fetch_online=False)
        country = meta.get('country', 'Unknown')
        
        signal = r.get('signal_type', 'HOLD').upper()
        if signal == 'BUY':
            country_stats[country]['buy'] += 1
        elif signal == 'SELL':
            country_stats[country]['sell'] += 1
        else:
            country_stats[country]['hold'] += 1
        
        country_stats[country]['confidences'].append(r.get('confidence', 0.5))
    
    output = {}
    for country, stats in country_stats.items():
        confs = stats['confidences']
        output[country] = {
            'buy': stats['buy'],
            'sell': stats['sell'],
            'hold': stats['hold'],
            'total': stats['buy'] + stats['sell'] + stats['hold'],
            'buy_pct': stats['buy'] / max(1, stats['buy'] + stats['sell'] + stats['hold']),
            'avg_conf': sum(confs) / len(confs) if confs else 0.5
        }
    
    return dict(sorted(output.items(), key=lambda x: x[1]['total'], reverse=True))


def get_sector_heatmap_md(sector_agg: dict) -> str:
    """Generates markdown table for sector heatmap."""
    lines = ["### 🏭 Sector Heatmap", "| Sector | BUY % | Signals | Avg Conf |", "|--------|-------|---------|----------|"]
    
    for sector, stats in sector_agg.items():
        emoji = "🟢" if stats['buy_pct'] > 0.5 else ("🔴" if stats['buy_pct'] < 0.2 else "🟡")
        lines.append(f"| {emoji} {sector} | {stats['buy_pct']:.0%} | {stats['total']} | {stats['avg_conf']:.2f} |")
    
    return "\n".join(lines)


def get_country_summary_md(country_agg: dict) -> str:
    """Generates markdown table for country summary."""
    lines = ["### 🌍 Country Exposure", "| Country | BUY | SELL | HOLD | Total |", "|---------|-----|------|------|-------|"]
    
    for country, stats in country_agg.items():
        if stats['total'] > 0:
            lines.append(f"| {country} | {stats['buy']} | {stats['sell']} | {stats['hold']} | {stats['total']} |")
    
    return "\n".join(lines)


def get_twelve_data_metadata(ticker: str) -> dict:
    """Fetches company profile from Twelve Data."""
    import requests
    from config import TWELVE_DATA_API_KEY
    
    if not TWELVE_DATA_API_KEY:
        return {'sector': 'Unknown'}
        
    url = f"https://api.twelvedata.com/profile?symbol={ticker}&apikey={TWELVE_DATA_API_KEY}"
    
    try:
        response = requests.get(url, timeout=3) # Fast timeout
        if response.status_code == 200:
            data = response.json()
            # Twelve Data returns: sector, industry, country, etc.
            if 'sector' in data:
                 return {
                    'ticker': ticker,
                    'sector': data.get('sector', 'Unknown'),
                    'industry': data.get('industry', 'Unknown'),
                    'country': data.get('country', 'Unknown'),
                    'market_cap': 0 
                }
    except Exception:
        pass
        
    return {'sector': 'Unknown'}
