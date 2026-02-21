import requests
import pandas as pd
from bs4 import BeautifulSoup


def _get_universe_denyset() -> set:
    """Load configured ticker denylist for universe filtering."""
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from config import UNIVERSE_DENYLIST
    except Exception:
        return set()
    return {str(t).upper() for t in (UNIVERSE_DENYLIST or []) if str(t).strip()}


def get_sp500_tickers():
    """Fetches the list of S&P 500 tickers from Wikipedia."""
    print("Fetching S&P 500 ticker list...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        if table is None:
            return []
        tickers = []
        for row in table.find_all('tr')[1:]:
            ticker = row.find_all('td')[0].text.strip()
            tickers.append(ticker)
        # Clean tickers (replace . with - for yfinance compatibility)
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500: {e}")
        return []

def get_etf_universe():
    """Returns a comprehensive list of major ETFs across categories."""
    return {
        # Core Index ETFs (most liquid)
        'Index': [
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'IVV', 'RSP',
            'SPLG', 'QQQM', 'SPTM', 'ITOT', 'SCHB', 'SCHX',  # Alternatives
        ],
        # Size/Market Cap
        'Size': [
            'MDY', 'IJH', 'IWM', 'VB', 'IJR', 'VXF',  # Original
            'VO', 'IWR', 'IVOO', 'SCHA', 'VIOO', 'VBK', 'VBR',  # Mid/Small
            'IWO', 'IWN', 'SLYV', 'SLYG',  # Small Value/Growth
        ],
        # SPDR Sector ETFs (Complete XL series)
        'Sector': [
            'XLK', 'XLV', 'XLF', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC',
        ],
        # Vanguard Sector ETFs
        'Vanguard_Sector': [
            'VGT', 'VHT', 'VFH', 'VCR', 'VDC', 'VDE', 'VIS', 'VAW', 'VPU', 'VOX',
        ],
        # iShares Industry ETFs
        'Industry': [
            'IBB', 'IGV', 'SOXX', 'ITA', 'IHI', 'IYT', 'IYW', 'IYF', 'IYH', 'IYE',
            'XBI', 'XOP', 'XHB', 'XME', 'XSD', 'XSW', 'KRE', 'KBE', 'KIE',
            'SMH', 'XAR', 'JETS', 'PBW',  # Semis, Aerospace, Airlines, Clean Energy
        ],
        # Thematic/Innovation ETFs
        'Thematic': [
            'ARKK', 'ARKW', 'ARKG', 'ARKF', 'ARKQ',  # ARK suite
            'BOTZ', 'ROBO', 'HACK', 'SKYY', 'FINX', 'GNOM',  # Original
            'KWEB', 'CIBR', 'CLOU', 'WCLD', 'IGF', 'LIT', 'TAN', 'ICLN',  # Tech/Clean
            'BLOK', 'VPN', 'DRIV', 'IDRV', 'KARS',  # Blockchain, EV
        ],
        # Dividend/Income ETFs
        'Dividend': [
            'VIG', 'VYM', 'DVY', 'SCHD', 'SDY', 'NOBL', 'DGRO',  # Original
            'SPYD', 'HDV', 'SPHD', 'DIVO', 'JEPI', 'JEPQ',  # High Yield
            'RDIV', 'FVD', 'DLN', 'DHS',  # Deep Value Dividend
        ],
        # Factor/Smart Beta ETFs
        'Factor': [
            'VUG', 'IWF', 'MTUM', 'QUAL', 'IUSV',  # Original Growth
            'VTV', 'IWD', 'VLUE', 'RPV',  # Original Value
            'USMV', 'SPLV', 'EFAV', 'EEMV',  # Low Volatility
            'SIZE', 'SMMV', 'XSVM',  # Size Factor
            'MOAT', 'PKW',  # Wide Moat, Buybacks
        ],
        # International Developed
        'International_Dev': [
            'EFA', 'VGK', 'EWJ', 'EWG', 'EWU', 'EWQ', 'EWP', 'EWI', 'EWL', 'EWA',
            'EWC', 'EWS', 'EWH', 'EWT', 'EWY', 'EWN', 'EWD', 'EWK', 'EWO',
            'IEFA', 'SCHF', 'VEA', 'SPDW', 'HEFA',  # Broad Developed
        ],
        # Emerging Markets
        'Emerging': [
            'EEM', 'VWO', 'IEMG', 'INDA', 'EWZ', 'FXI', 'MCHI', 'KWEB',
            'EPOL', 'EZA', 'EWW', 'EWT', 'TUR', 'THD', 'EIDO', 'EPU', 'ECH',
            'RSX', 'ERUS', 'GXG', 'ARGT', 'FM', 'VNM',  # Single Country/Frontier
        ],
        # Commodity ETFs
        'Commodity': [
            'GLD', 'IAU', 'SLV', 'USO', 'DBA', 'PDBC', 'DBC', 'COPX', 'UNG',
            'GDX', 'GDXJ', 'SIL', 'PALL', 'PPLT', 'WEAT', 'CORN', 'SOYB',
            'URA', 'URNM',  # Uranium
            'REMX',  # Rare Earth
        ],
        # Real Estate ETFs
        'Real_Estate': [
            'VNQ', 'IYR', 'XLRE', 'RWR', 'REET',
            'SCHH', 'USRT', 'REM', 'MORT', 'VNQI', 'RWX',  # REITs International
        ],
        # Leveraged/Inverse (Use sparingly)
        'Leveraged': [
            'TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'SOXL', 'SOXS',
            'TNA', 'TZA', 'LABU', 'LABD', 'FAS', 'FAZ',
        ],
        # Volatility
        'Volatility': ['VIXY', 'UVXY', 'SVXY', 'VXX'],
        # Crypto
        'Crypto': ['IBIT', 'FBTC', 'BITO', 'GBTC', 'ETHE', 'FDIG'],
    }

def get_bond_universe():
    """Returns a comprehensive list of Bond/Fixed Income ETFs."""
    return {
        'Total_Market': ['AGG', 'BND', 'BNDX', 'SCHZ', 'FBND', 'NUAG'],
        # Treasury - by duration
        'Treasury_Short': ['SHY', 'BIL', 'VGSH', 'SCHO', 'SGOV', 'GBIL', 'CLIP'],
        'Treasury_Mid': ['IEF', 'VGIT', 'GOVT', 'SCHR', 'SPTL'],
        'Treasury_Long': ['TLT', 'VGLT', 'SPTL', 'EDV', 'ZROZ'],
        'Treasury_Floating': ['USFR', 'TFLO', 'FLOT'],
        # Corporate
        'Corporate_IG': ['LQD', 'VCIT', 'VCSH', 'IGIB', 'USIG', 'IGSB', 'SLQD'],
        'High_Yield': ['HYG', 'JNK', 'SHYG', 'USHY', 'HYLS', 'SJNK', 'HYLB'],
        # Ultrashort/Money Market alternatives
        'Ultrashort': ['MINT', 'ICSH', 'NEAR', 'JPST', 'GSY', 'VNLA', 'FLRN'],
        # Inflation Protected
        'Inflation': ['TIP', 'VTIP', 'SCHP', 'STIP', 'TDTT'],
        # Municipal
        'Muni': ['MUB', 'VTEB', 'TFI', 'SUB', 'HYD', 'SHYD', 'CMF'],
        # International/EM Bonds
        'International': ['EMB', 'VWOB', 'IAGG', 'BNDW', 'PCY', 'EMHY'],
    }

def get_priority_from_db(top_n: int = 50):
    """Dynamically selects high-impact tickers based on average volume in the DB."""
    try:
        from storage import get_connection
        con = get_connection()
        # Get tickers with highest average volume
        query = """
            SELECT ticker, AVG(volume) as avg_vol
            FROM price_history
            GROUP BY ticker
            HAVING COUNT(*) > 200
            ORDER BY avg_vol DESC
            LIMIT ?
        """
        result = con.execute(query, [top_n]).fetchall()
        con.close()
        if result:
            return [r[0] for r in result]
    except Exception as e:
        print(f"Warning: Could not get priority from DB: {e}")
    return []

def get_prioritized_universe():
    """Returns the universe sorted by priority (Holdings > Watchlist > DB volume > ETFs > Bonds > S&P 500)."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import PORTFOLIO_HOLDINGS, WATCHLIST, PRIORITIZE_HOLDINGS, SAFE_ASSET_BENCHMARK_TICKER
    
    # Priority Order: Holdings > Watchlist > DB High-Volume > ETFs > Bonds > S&P 500
    priority = []
    seen = set()

    # -1. Safe asset benchmark first (if set)
    if SAFE_ASSET_BENCHMARK_TICKER:
        bench = SAFE_ASSET_BENCHMARK_TICKER
        if bench and bench not in seen:
            priority.append(bench)
            seen.add(bench)

    # 0. Portfolio Holdings FIRST (if enabled)
    if PRIORITIZE_HOLDINGS:
        for t in PORTFOLIO_HOLDINGS:
            if t and t not in seen:
                priority.append(t)
                seen.add(t)
        for t in WATCHLIST:
            if t and t not in seen:
                priority.append(t)
                seen.add(t)
    
    # 1. Try to get high-volume tickers from DB
    db_priority = get_priority_from_db(50)
    
    # 2. Static fallbacks
    etfs = []
    for cat in get_etf_universe().values():
        etfs.extend(cat)
    
    bonds = []
    for cat in get_bond_universe().values():
        bonds.extend(cat)
        
    sp500 = get_sp500_tickers()
    
    for t in db_priority + etfs + bonds + sp500:
        if t not in seen:
            priority.append(t)
            seen.add(t)
            
    denyset = _get_universe_denyset()
    if not denyset:
        return priority
    return [t for t in priority if t.upper() not in denyset]

def get_nasdaq_100_tickers():
    """Fetches the list of Nasdaq 100 tickers from Wikipedia."""
    print("Fetching Nasdaq 100 ticker list...")
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # The table id for Nasdaq 100 might be different or just the first table
        table = soup.find('table', {'id': 'constituents'})
        if not table:
             # Fallback if id is not found (often it's just 'constituents' but wikipedia changes)
             tables = soup.find_all('table')
             for t in tables:
                 if "Ticker" in t.text and "Company" in t.text:
                     table = t
                     break
        
        tickers = []
        if table:
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if cols:
                    ticker = cols[0].text.strip()
                    tickers.append(ticker)
        
        # Clean tickers
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception as e:
        print(f"Error fetching Nasdaq 100: {e}")
        return []

def get_international_adrs():
    """Returns a curated list of high-volume International ADRs (verified 2024-2025)."""
    return [
        # Technology / Semiconductors
        'TSM', 'ASML', 'INFY', 'WIT', 'UMC', 'ASX', 'STM', 'NXPI', 'SAP', 'SHOP',
        'SE', 'GRAB', 'CPNG',  # SE Asia Tech
        # Pharma / Healthcare
        'NVO', 'AZN', 'SNY', 'NVS', 'GSK', 'TAK', 'RHHBY', 'BAYRY', 'ABBV',
        # Resources / Energy / Materials
        'BHP', 'RIO', 'VALE', 'PBR', 'SHEL', 'TTE', 'BP', 'E', 'SU', 'CNQ',
        'SCCO', 'FCX', 'GOLD', 'NEM', 'FNV', 'WPM',  # Mining/Precious
        # China / Hong Kong (High Volume)
        'BABA', 'PDD', 'JD', 'BIDU', 'TCEHY', 'NIO', 'XPEV', 'LI',
        'NTES', 'BILI', 'TME', 'KC', 'ZTO', 'YUMC', 'TCOM', 'VNET',
        # India (Growth Market)
        'HDB', 'IBN', 'INFY', 'WIT', 'SIFY', 'RDY', 'WNS', 'TTM',
        # Latin America
        'MELI', 'NU', 'STNE', 'PAGS',  # Fintech
        'ITUB', 'BBD', 'ABEV', 'SBS', 'CIG', 'BVN',  # Brazil/Peru
        # Europe Finance
        'BCS', 'LYG', 'HSBC', 'UBS', 'CS', 'DB', 'ING', 'SAN', 'BBVA',
        # Japan
        'SONY', 'HMC', 'TM', 'MUFG', 'SMFG', 'MFG', 'NMR', 'NTDOY',
        # Consumer / Auto / Industrial
        'UL', 'BUD', 'DEO', 'BTI', 'PM', 'STZ',  # Consumer Staples
        'ABB', 'CAJ', 'FANUY',  # Industrials
        'RELX', 'TRI', 'WPP',  # Media/Info
    ]

def get_master_universe():
    """Combines all tickers into a single list."""
    # Ensure priority logic is kept
    priority_list = get_prioritized_universe()
    
    # Add new sources
    nasdaq = get_nasdaq_100_tickers()
    adrs = get_international_adrs()
    
    # Merge and deduplicate, keeping priority order
    seen = set(priority_list)
    final_list = list(priority_list)
    
    for t in nasdaq + adrs:
        if t not in seen:
            final_list.append(t)
            seen.add(t)
            
    denyset = _get_universe_denyset()
    if not denyset:
        return final_list
    return [t for t in final_list if t.upper() not in denyset]

# Path to scan state file for LRU tracking
import os as _os
SCAN_STATE_FILE = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'data', 'scan_state.json')

def get_lru_universe(prioritize_portfolio: bool = True) -> list:
    """
    Returns universe ordered by LRU (least recently updated first).
    Portfolio/watchlist tickers go first if prioritize_portfolio=True.
    Tickers never scanned are placed after priority tickers but before recently scanned.
    """
    import sys, os, json
    from datetime import datetime
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import PORTFOLIO_HOLDINGS, WATCHLIST
    
    # Get the base universe (all tickers, deduplicated)
    base_universe = get_master_universe()
    
    # Load scan state
    last_scanned = {}
    try:
        if os.path.exists(SCAN_STATE_FILE):
            with open(SCAN_STATE_FILE, 'r') as f:
                state = json.load(f)
            last_scanned = state.get('last_scanned', {})
    except Exception as e:
        print(f"Warning: Could not load scan state: {e}")
    
    # Separate tickers into categories
    priority_set = set(t.upper() for t in (PORTFOLIO_HOLDINGS + WATCHLIST))
    priority_tickers = []
    other_tickers = []
    
    for ticker in base_universe:
        ticker_u = ticker.upper()
        if prioritize_portfolio and ticker_u in priority_set:
            priority_tickers.append(ticker)
        else:
            other_tickers.append(ticker)
    
    # Sort other tickers by last_scanned timestamp (oldest first, never-scanned at top)
    def get_timestamp(ticker):
        ts = last_scanned.get(ticker.upper())
        if not ts:
            return datetime.min  # Never scanned = oldest
        try:
            return datetime.fromisoformat(ts)
        except:
            return datetime.min
    
    other_tickers.sort(key=get_timestamp)
    
    # If prioritizing portfolio, sort priority tickers by timestamp too
    if prioritize_portfolio:
        priority_tickers.sort(key=get_timestamp)
        return priority_tickers + other_tickers
    else:
        # Pure LRU - all tickers sorted by timestamp
        all_tickers = priority_tickers + other_tickers
        all_tickers.sort(key=get_timestamp)
        return all_tickers

if __name__ == "__main__":
    # Test
    tickers = get_master_universe()
    print(f"Total tickers in universe: {len(tickers)}")
    print(f"Sample tickers: {tickers[:20]}")
