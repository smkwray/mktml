"""
MktML Configuration
===================
Centralized configuration for all tunable parameters.
Edit this file to adjust behavior without modifying source code.

Public template notes:
- This example is sanitized for open-source publishing.
- Copy to `config.py` for local use and supply credentials via environment variables.
"""
import json
import os
import sys
from typing import Dict, Optional

# =============================================================================
# PATH DEFINITIONS
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "meta_classifier.joblib")
FEEDBACK_PATH = os.path.join(MODEL_DIR, "feedback_weights.joblib")
MODEL_THRESHOLDS_FILE = os.path.join(MODEL_DIR, "model_thresholds.json")
MODEL_MANIFEST_FILE = os.path.join(MODEL_DIR, "manifest.json")
CALIBRATION_ARTIFACT_DIR = os.path.join(MODEL_DIR, "calibration")
CALIBRATION_BINS = 16
CALIBRATION_MIN_SAMPLES = 300
PROB_CALIBRATION_FILE_TEMPLATE = "probability_calibration_h{horizon}d.json"

DEFAULT_BUY_THRESHOLD = 0.60
DEFAULT_SELL_THRESHOLD = 0.40

# =============================================================================
# THREADING & RESOURCE MANAGEMENT
# =============================================================================
# Reserve headroom on any machine by default: usable_cores = total_cores - reserved_cores.
# Override with:
#   CPU_RESERVED_CORES=<int>  (default 4)
#   ML_N_JOBS=<int>
#   SCANNER_WORKERS=<int>
TOTAL_CPU_CORES = os.cpu_count() or 8
RESERVED_CPU_CORES = int(os.environ.get('CPU_RESERVED_CORES', 4))
if RESERVED_CPU_CORES < 0:
    RESERVED_CPU_CORES = 0
if RESERVED_CPU_CORES >= TOTAL_CPU_CORES:
    RESERVED_CPU_CORES = max(0, TOTAL_CPU_CORES - 1)
AVAILABLE_CPU_CORES = max(1, TOTAL_CPU_CORES - RESERVED_CPU_CORES)

# Main ML parallelism budget (RF/XGB n_jobs and related training compute).
N_JOBS = int(os.environ.get('ML_N_JOBS', AVAILABLE_CPU_CORES))
if N_JOBS < 1:
    N_JOBS = 1
N_JOBS = min(N_JOBS, TOTAL_CPU_CORES)

# Scanner workers are used directly in scanner/data-loader and multiplied in ML feature extraction.
# Default keeps feature extraction near N_JOBS because ml_engine uses SCANNER_WORKERS * 4.
_default_scanner_workers = max(1, (N_JOBS + 3) // 4)
SCANNER_WORKERS = int(os.environ.get('SCANNER_WORKERS', _default_scanner_workers))
if SCANNER_WORKERS < 1:
    SCANNER_WORKERS = 1

# macOS-specific thread controls (prevent oversubscription in BLAS/OpenMP)
if sys.platform == "darwin":
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')

# =============================================================================
# ML MODEL PARAMETERS
# =============================================================================
# Training Mode: 'FAST' (quick iteration) or 'SLOW' (thorough, for production)
ML_MODE = 'SLOW'  # Options: 'FAST', 'SLOW'

# Toggle XGBoost (set to False if causing issues).
# Env override allows one-off retrains without changing committed defaults:
# ENABLE_XGBOOST=0 .venv/bin/python src/main.py --train-ml
_enable_xgb_env = str(os.environ.get('ENABLE_XGBOOST', '1')).strip().lower()
ENABLE_XGBOOST = _enable_xgb_env not in {'0', 'false', 'no', 'off'}

# Mode-specific parameters
if ML_MODE == 'FAST':
    # FAST mode: quick iteration, good for testing
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 8
    RF_MIN_SAMPLES_LEAF = 30
    
    GBM_N_ESTIMATORS = 75
    GBM_MAX_DEPTH = 4
    GBM_MIN_SAMPLES_LEAF = 30
    
    XGB_N_ESTIMATORS = 100
    XGB_MAX_DEPTH = 5
    XGB_LEARNING_RATE = 0.15
    XGB_SUBSAMPLE = 0.8
    XGB_COLSAMPLE = 0.8
    
    CV_SPLITS = 2  # Minimal cross-validation
    
else:  # SLOW mode
    # SLOW mode: thorough training for production
    RF_N_ESTIMATORS = 300
    RF_MAX_DEPTH = 15
    RF_MIN_SAMPLES_LEAF = 20
    
    GBM_N_ESTIMATORS = 200
    GBM_MAX_DEPTH = 8
    GBM_MIN_SAMPLES_LEAF = 20
    
    XGB_N_ESTIMATORS = 300
    XGB_MAX_DEPTH = 8
    XGB_LEARNING_RATE = 0.05  # Lower LR = more iterations needed but better
    XGB_SUBSAMPLE = 0.85
    XGB_COLSAMPLE = 0.85
    
    CV_SPLITS = 5  # More thorough cross-validation

# Ensemble weights (XGBoost gets slight preference based on research)
ENSEMBLE_WEIGHTS = [1.0, 1.0, 1.5]  # RF, GBM, XGB

# =============================================================================
# SIGNAL & STRATEGY PARAMETERS
# =============================================================================
# RSI thresholds
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_EXIT = 80

# Signal persistence (days of consecutive signal required)
SIGNAL_PERSISTENCE_DAYS = 2

# High-confidence threshold for reporting
HIGH_CONFIDENCE_THRESHOLD = 0.6

# Signal Smoothing (reduces whiplash from single-day price moves)
# When enabled, confidence scores are EMA-smoothed with previous values
ENABLE_SIGNAL_SMOOTHING = True  # Toggle smoothing on/off
SIGNAL_SMOOTHING_ALPHA = 0.3    # 0.0-1.0: Lower = more smoothing (0.3 = 70% weight to previous)

# =============================================================================
# MULTI-HORIZON ML STRATEGY
# =============================================================================
# Prediction horizons in trading days
HORIZONS = [5, 10, 30]

# Model paths for each horizon
MODEL_PATH_5D = os.path.join(MODEL_DIR, 'model_5d.pkl')
MODEL_PATH_10D = os.path.join(MODEL_DIR, 'model_10d.pkl')
MODEL_PATH_30D = os.path.join(MODEL_DIR, 'model_30d.pkl')

# Asset-bucket model strategy
# Train a global model plus per-bucket models to reduce cross-regime noise.
ASSET_BUCKET_MODELING_ENABLED = True
ASSET_BUCKETS = ['EQUITY', 'ETF', 'BOND']
ASSET_BUCKET_MIN_TICKERS = 25
ASSET_BUCKET_MIN_SAMPLES = 1500
BUCKET_MODEL_DIR = os.path.join(MODEL_DIR, 'buckets')

# Strict walk-forward validation (out-of-sample only)
WALK_FORWARD_ENABLED = True
WALK_FORWARD_SPLITS = 4
WALK_FORWARD_TOP_QUANTILE = 0.90
WALK_FORWARD_MIN_TEST_SAMPLES = 200
WALK_FORWARD_GROUP_BY_DATE = True
WALK_FORWARD_TOP_K_FRAC = 0.10
WALK_FORWARD_TOP_K_MIN_SAMPLES = 50
MODEL_VALIDATION_FILE = os.path.join(PROJECT_ROOT, 'reports', 'model_validation.json')

# Target returns per horizon (used for Triple Barrier)
HORIZON_TARGETS = {
    5: 0.015000,
    10: 0.030000,
    30: 0.080000,
}

# Confluence thresholds
CONFLUENCE_THRESHOLD = 0.6
TACTICAL_THRESHOLD = 0.7
TREND_THRESHOLD = 0.65

# Dividend feature parity
# False = basic dividend features only (matches scan inference defaults)
USE_ADVANCED_DIVIDEND_FEATURES = False

# Recency weighting (training)
RECENCY_WEIGHTING_ENABLED = True
RECENCY_HALF_LIFE_DAYS = 365
RECENCY_MIN_WEIGHT = 0.10

def _load_thresholds_file() -> dict:
    try:
        if os.path.exists(MODEL_THRESHOLDS_FILE):
            with open(MODEL_THRESHOLDS_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        return {}
    return {}


def get_model_thresholds(horizon: Optional[int] = None) -> Dict[str, float]:
    """Return calibrated thresholds for a horizon, falling back to defaults."""
    defaults = {'buy': DEFAULT_BUY_THRESHOLD, 'sell': DEFAULT_SELL_THRESHOLD}
    data = _load_thresholds_file()
    if horizon is None:
        return data
    entry = data.get(str(horizon), {})
    if not isinstance(entry, dict):
        return defaults
    try:
        buy = float(entry.get('buy', defaults['buy']))
    except (TypeError, ValueError):
        buy = defaults['buy']
    try:
        sell = float(entry.get('sell', defaults['sell']))
    except (TypeError, ValueError):
        sell = defaults['sell']
    return {'buy': buy, 'sell': sell}

# =============================================================================
# ML FEATURE CONTRACT (STRICT ORDERING)
# =============================================================================
# CRITICAL: This list MUST match the exact alphabetical order of features
# used during model training. Adding/removing items here will break inference
# unless models are retrained.
ML_FEATURE_CONTRACT = [
    'atr_14', 'atr_ratio', 'bb_lower', 'bb_middle', 'bb_pct_b', 'bb_position', 'bb_upper',
    'day_of_week', 'day_of_year', 'days_to_ex_div_norm', 'dist_52w_high', 'dist_52w_low',
    'dist_sma_200', 'dist_sma_50', 'dividend_growth_5y', 'dividend_safety_score', 'dividend_yield',
    'high_close', 'high_low', 'inter_vix_rsi', 'inter_vix_sma200', 'inter_yield_rsi', 'inter_yield_sma200',
    'is_dividend_stock', 'is_month_end', 'is_month_start', 'is_yield_trap', 'low_close',
    'macd', 'macd_hist_slope', 'macd_histogram', 'macd_signal', 'macro_fed_assets_chg', 'macro_hy_spread', 'macro_stress',
    'macro_vix', 'macro_yield_curve', 'month', 'payout_quality', 'price_vs_52w_high',
    # Qualitative features (ordinal)
    'qual_confidence', 'qual_cyclical', 'qual_debt', 'qual_is_etf', 'qual_maturity', 'qual_moat',
    # Qualitative features (industry one-hot)
    'qual_industry_aerospace_and_defense', 'qual_industry_asset_management', 'qual_industry_automotive',
    'qual_industry_banks', 'qual_industry_biotech', 'qual_industry_bond_etf', 'qual_industry_chemicals',
    'qual_industry_construction_materials', 'qual_industry_diversified_etf', 'qual_industry_energy_equipment',
    'qual_industry_food_and_beverage', 'qual_industry_hardware_and_semiconductors', 'qual_industry_healthcare_equipment',
    'qual_industry_healthcare_services', 'qual_industry_hotels_and_leisure', 'qual_industry_household_products',
    'qual_industry_industrial_machinery', 'qual_industry_insurance', 'qual_industry_international_etf',
    'qual_industry_it_services', 'qual_industry_media_and_entertainment', 'qual_industry_metals_and_mining',
    'qual_industry_oil_and_gas', 'qual_industry_pharmaceuticals', 'qual_industry_real_estate_services',
    'qual_industry_reits', 'qual_industry_retail', 'qual_industry_sector_etf', 'qual_industry_software',
    'qual_industry_telecom', 'qual_industry_transportation', 'qual_industry_unknown', 'qual_industry_utilities',
    # Qualitative features (sector one-hot)
    'qual_sector_communication_services', 'qual_sector_consumer_discretionary', 'qual_sector_consumer_staples',
    'qual_sector_diversified', 'qual_sector_energy', 'qual_sector_financials', 'qual_sector_healthcare',
    'qual_sector_industrials', 'qual_sector_materials', 'qual_sector_real_estate', 'qual_sector_technology',
    'qual_sector_unknown', 'qual_sector_utilities',
    # Continued features
    'returns_10d', 'returns_1d', 'returns_20d', 'returns_5d', 'rsi', 'rsi_overbought',
    'rsi_oversold', 'rsi_slope', 'sma_200', 'sma_50', 'sma_50_slope', 'sma_cross_dist', 'stoch_d', 'stoch_k',
    'stoch_overbought', 'stoch_oversold', 'tr', 'volatility_20d', 'volatility_regime',
    'volume_sma_20', 'volume_trend', 'volume_zscore', 'week_of_year'
]

# =============================================================================
# QUALITATIVE FEATURE SETTINGS
# =============================================================================
# Configurable via dashboard (saved to data/qual_settings.json)
QUAL_UPDATE_ENABLED = True
QUAL_UPDATE_CHUNK_SIZE = 20      # Tickers per scheduled run
QUAL_UPDATE_INTERVAL_HOURS = 4   # Hours between runs
QUAL_FEATURES_FILE = os.path.join(PROJECT_ROOT, 'data', 'qualitative_features.json')

# Encoding maps for ordinal features
QUAL_MATURITY_MAP = {"startup": 0, "growth": 1, "mature": 2, "declining": 3}
QUAL_CYCLICAL_MAP = {"defensive": 0, "mixed": 1, "cyclical": 2}
QUAL_MOAT_MAP = {"none": 0, "narrow": 1, "wide": 2}
QUAL_DEBT_MAP = {"low": 0, "medium": 1, "high": 2}

# Sector one-hot columns (GICS + Diversified for ETFs + Unknown)
QUAL_SECTOR_COLUMNS = [
    'qual_sector_communication_services',
    'qual_sector_consumer_discretionary',
    'qual_sector_consumer_staples',
    'qual_sector_diversified',
    'qual_sector_energy',
    'qual_sector_financials',
    'qual_sector_healthcare',
    'qual_sector_industrials',
    'qual_sector_materials',
    'qual_sector_real_estate',
    'qual_sector_technology',
    'qual_sector_unknown',
    'qual_sector_utilities',
]

# Industry one-hot columns (24 industries + ETF types + Unknown)
QUAL_INDUSTRY_COLUMNS = [
    'qual_industry_aerospace_and_defense',
    'qual_industry_asset_management',
    'qual_industry_automotive',
    'qual_industry_banks',
    'qual_industry_biotech',
    'qual_industry_bond_etf',
    'qual_industry_chemicals',
    'qual_industry_construction_materials',
    'qual_industry_diversified_etf',
    'qual_industry_energy_equipment',
    'qual_industry_food_and_beverage',
    'qual_industry_hardware_and_semiconductors',
    'qual_industry_healthcare_equipment',
    'qual_industry_healthcare_services',
    'qual_industry_hotels_and_leisure',
    'qual_industry_household_products',
    'qual_industry_industrial_machinery',
    'qual_industry_insurance',
    'qual_industry_international_etf',
    'qual_industry_it_services',
    'qual_industry_media_and_entertainment',
    'qual_industry_metals_and_mining',
    'qual_industry_oil_and_gas',
    'qual_industry_pharmaceuticals',
    'qual_industry_real_estate_services',
    'qual_industry_reits',
    'qual_industry_retail',
    'qual_industry_sector_etf',
    'qual_industry_software',
    'qual_industry_telecom',
    'qual_industry_transportation',
    'qual_industry_unknown',
    'qual_industry_utilities',
]

# =============================================================================
# DIVIDEND STOCK SETTINGS
# =============================================================================
# Tickers classified as dividend/income focused (more lenient sell signals)
DIVIDEND_TICKERS = [
    # 'TICKER_A',
    # 'TICKER_B',
]

# Dividend stocks get more lenient exit signals (hold through minor dips)
DIVIDEND_RSI_EXIT = 85       # Higher RSI threshold before sell (vs 80 for growth)
DIVIDEND_SELL_BUFFER = 0.02  # Require 2% below SMA200 before sell (growth = 0%)

# Definition of "Safe Asset" (for separate reporting)
SAFE_ASSET_ATR_THRESHOLD = 0.005  # ATR < 0.5% considered safe/low vol (Bond/MM like)
SAFE_ASSET_BENCHMARK_TICKER = None  # Deprecated; yield benchmark is now fixed
SAFE_ASSET_BENCHMARK_YIELD_FALLBACK = None  # Deprecated; no longer used
SAFE_ASSET_BENCHMARK_YIELD_7D = 0.0348  # 7-day yield benchmark (as decimal)
SAFE_ASSET_MIN_YIELD_MULTIPLIER = 1.0  # Require yield >= benchmark yield * multiplier
SAFE_ASSET_RETURN_DAYS = 30
SAFE_ASSET_MIN_RETURN_ABS = 0.0  # Require 30d return >= this absolute floor
SAFE_ASSET_TRAILING_DIVIDEND_DAYS = 365
SAFE_ASSET_MAX_DIVIDEND_AGE_DAYS = 180
SAFE_ASSET_ALLOWLIST = [
    # Core Bond ETFs
    'VCRB',   # Vanguard Core Bond ETF
    'CVSB',   # Calvert Ultra-Short Investment
    'BND',    # Vanguard Total Bond Market
    'AGG',    # iShares Core U.S. Aggregate Bond
    'SCHZ',   # Schwab U.S. Aggregate Bond
    
    # Treasury - Ultra Short (0-3 months) - Highest Yield
    'SGOV',   # iShares 0-3 Month Treasury Bond ETF (TOP PICK)
    'BIL',    # SPDR Bloomberg 1-3 Month T-Bill
    'CLIP',   # Global X 1-3 Month T-Bill ETF
    'GBIL',   # Goldman Sachs Access Treasury 0-1 Year ETF
    
    # Treasury - Short Term (1-3 years)
    'SHY',    # iShares 1-3 Year Treasury Bond
    'VGSH',   # Vanguard Short-Term Treasury
    'SCHO',   # Schwab Short-Term US Treasury ETF
    
    # Treasury - Floating Rate (adjusts with rates)
    'USFR',   # WisdomTree Floating Rate Treasury Fund
    'TFLO',   # iShares Treasury Floating Rate Bond ETF
    'FLOT',   # iShares Floating Rate Bond ETF (~4.8% yield)
    'FLRN',   # SPDR Bloomberg Floating Rate Bond (near-zero duration, ~5% yield)
    
    # TIPS - Inflation Protected Short Duration
    'STIP',   # iShares 0-5 Year TIPS Bond ETF (very low vol, 4%+ yield)
    
    # Treasury - Intermediate
    'VGIT',   # Vanguard Intermediate-Term Treasury
    'IEF',    # iShares 7-10 Year Treasury Bond
    'GOVT',   # iShares U.S. Treasury Bond
    
    # Ultrashort Active (Higher Yield Potential)
    'MINT',   # PIMCO Enhanced Short Maturity Active ETF
    'ICSH',   # iShares Ultra Short-Term Bond
    'NEAR',   # iShares Short Maturity Bond
    'JPST',   # JPMorgan Ultra-Short Income ETF
    'GSY',    # Invesco Ultra Short Duration ETF
    'VNLA',   # Janus Henderson Short Duration Income ETF
    
    # Corporate Bond - Short Term
    'VCIT',   # Vanguard Intermediate-Term Corporate Bond
    'VCSH',   # Vanguard Short-Term Corporate Bond
    'USIG',   # iShares Broad USD Investment Grade Corporate
    
    # High Yield (Higher Risk/Reward)
    'JNK',    # SPDR Bloomberg High Yield Bond
    'HYG',    # iShares High Yield Corporate Bond
    
    # Municipal (Tax Advantaged)
    'MUB',    # iShares National Muni Bond
    'VTEB',   # Vanguard Tax-Exempt Bond
    'TFI',    # SPDR Nuveen Bloomberg Municipal Bond
    
    # Inflation Protected
    'TIP',    # iShares TIPS Bond
]
SAFE_ASSET_DENYLIST = [
    # Explicit equities or misclassified tickers to exclude from safe assets
]

# External yields file (auto-populated by gemini-cli, see scripts/update_yields.py)
SAFE_ASSET_YIELDS_FILE = os.path.join(PROJECT_ROOT, 'data', 'safe_asset_yields.json')

# =============================================================================
# SCANNING PARAMETERS
# =============================================================================
# Batch size for downloading tickers
BATCH_SIZE = 20

# Rate limit retry settings
MAX_RETRIES = 3
BASE_RETRY_DELAY = 30  # seconds
MAX_RETRY_DELAY = 300  # seconds

# Scan recovery: hours before portfolio is "stale" and reprioritized on recovery
# If last scan failed and portfolio was scanned within this window, use pure LRU
PORTFOLIO_STALENESS_HOURS = 12
# Price data older than this is marked stale in scan/report health metrics.
MAX_PRICE_DATA_STALENESS_DAYS = 3
# If False, stale ticker data is excluded from recommendation generation (safer).
ALLOW_STALE_DATA_FOR_SIGNALS = False
# Tradability / liquidity filters for BUY recommendations.
ENFORCE_TRADABILITY_FILTERS = True
TRADABILITY_MIN_PRICE = 3.0
TRADABILITY_MIN_AVG_VOLUME_20D = 300000
TRADABILITY_MIN_AVG_DOLLAR_VOLUME_20D = 5000000
TRADABILITY_MAX_ATR_RATIO = 0.2
# Optional explicit exemptions (always uppercased downstream).
TRADABILITY_EXEMPT_TICKERS = []

# =============================================================================
# API KEYS
# =============================================================================
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL = "https://data.alpaca.markets/v2"  # Paper/Live data endpoint

def _split_csv_env(value: str) -> list[str]:
    """Split comma/semicolon API key env var strings into cleaned entries."""
    if not value:
        return []
    out = []
    for raw in value.replace(";", ",").split(","):
        key = raw.strip()
        if key:
            out.append(key)
    return out


def _merge_unique(items: list[str]) -> list[str]:
    """Deduplicate while preserving order."""
    seen = set()
    out = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


ALPHA_VANTAGE_API_KEYS = _merge_unique(
    _split_csv_env(os.environ.get('ALPHA_VANTAGE_API_KEYS', ''))
    + [
        os.environ.get('ALPHA_VANTAGE_API_KEY', '').strip(),
        os.environ.get('ALPHA_VANTAGE_KEY', '').strip(),
        os.environ.get('ALPHA_VANTAGE_API_KEY_2', '').strip(),
    ]
)
ALPHA_VANTAGE_API_KEY = ALPHA_VANTAGE_API_KEYS[0] if ALPHA_VANTAGE_API_KEYS else ''
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', '')
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '')
TIINGO_API_KEY = os.environ.get('TIINGO_API_KEY', '')
TWELVE_DATA_API_KEY = os.environ.get('TWELVE_DATA_API_KEY', '')  # Free tier: 8 calls/min
FMP_API_KEY = os.environ.get('FMP_API_KEY', '').strip()  # Financial Modeling Prep
EODHD_API_KEY = os.environ.get('EODHD_API_KEY', '').strip()

# Gemini CLI last-ditch market-data fallback (used only if all APIs fail)
ENABLE_GEMINI_PRICE_FALLBACK = True
GEMINI_PRICE_PRIMARY_MODEL = "gemini-3-flash-preview"
GEMINI_PRICE_FALLBACK_MODEL = ""  # Empty -> default gemini-cli model
GEMINI_PRICE_BATCH_SIZE = 3
GEMINI_PRICE_TIMEOUT_SECONDS = 180
GEMINI_PRICE_MIN_BARS = 200

# =============================================================================
# DATA PARAMETERS
# =============================================================================
# Years of historical data to download
HISTORY_YEARS = 5

# Minimum data points required for analysis
MIN_DATA_POINTS = 250

# =============================================================================
# PORTFOLIO HOLDINGS
# =============================================================================
# Tickers to fully exclude from universe/scans/reports.
# Use uppercase symbols.
UNIVERSE_DENYLIST = [
    # e.g. 'TICKER_TO_EXCLUDE',
]

# Your current holdings - these get prioritized in scans and monitored for exits
PORTFOLIO_HOLDINGS = [
    # 'TICKER_A',
    # 'TICKER_B',
]

# Watchlist - tickers you're interested in buying
WATCHLIST = [
    # 'TICKER_X',
    # 'TICKER_Y',
]

# =============================================================================
# PORTFOLIO INTEGRATION SETTINGS
# =============================================================================
# Generate exit alerts for holdings (SELL signals on your positions)
MONITOR_EXIT_SIGNALS = True

# Prioritize holdings in scan order (scan these first)
PRIORITIZE_HOLDINGS = True

# Include holdings health summary in reports
INCLUDE_HOLDINGS_REPORT = True

# Include watchlist section in reports (separate from holdings)
INCLUDE_WATCHLIST_REPORT = True

# =============================================================================
# API KEYS (for external data sources)
# =============================================================================
# FRED API key - get free at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = os.environ.get("FRED_API_KEY", "").strip()

# =============================================================================
# MACRO DATA CONFIGURATION
# =============================================================================
# Cache duration in hours (to minimize API calls)
MACRO_CACHE_HOURS = 6  # Refresh macro data every 6 hours

# Key FRED series to fetch (tuned for market signals)
# Focus on high-frequency (daily/weekly) data for timely signals
FRED_SERIES = {
    # Employment (leading indicators)
    'ICSA': 'weekly',           # Initial Jobless Claims
    'UNRATE': 'monthly',        # Unemployment Rate
    'PAYEMS': 'monthly',        # Nonfarm Payrolls
    
    # Treasury Rates (daily - key for market timing)
    'DGS1MO': 'daily',          # 1-Month Treasury
    'DGS3MO': 'daily',          # 3-Month Treasury
    'DGS2': 'daily',            # 2-Year Treasury
    'DGS10': 'daily',           # 10-Year Treasury
    'DGS30': 'daily',           # 30-Year Treasury
    'T10Y2Y': 'daily',          # 2s10s Spread (yield curve inversion)
    'T5YIFR': 'daily',          # 5-Year Forward Inflation Expectation
    
    # Credit Spreads (daily - risk sentiment)
    'BAMLH0A0HYM2': 'daily',    # High Yield Spread (ICE BofA)
    'AAA10Y': 'daily',          # AAA Corporate - 10Y Treasury Spread
    'BAA10Y': 'daily',          # BAA Corporate - 10Y Treasury Spread
    
    # Volatility & Stress (daily)
    'VIXCLS': 'daily',          # VIX - Market Fear Index
    'STLFSI4': 'weekly',        # St. Louis Fed Financial Stress Index
    'ANFCI': 'weekly',          # Chicago Fed National Financial Conditions
    
    # Fed Balance Sheet (weekly)
    'WALCL': 'weekly',          # Fed Total Assets
    'TREAST': 'weekly',         # Fed Treasury Holdings
    'WSHOMCB': 'weekly',        # Fed MBS Holdings
    
    # Bank Data (weekly - credit conditions)
    'TOTBKCR': 'weekly',        # Total Bank Credit
    'BUSLOANS': 'weekly',       # Commercial & Industrial Loans
    'DPSACBW027SBOG': 'weekly', # Bank Deposits
    
    # Money Supply
    'M2SL': 'monthly',          # M2 Money Stock
    'BOGMBASE': 'monthly',      # Monetary Base
    
    # Consumer & Sentiment (monthly)
    'UMCSENT': 'monthly',       # U of Michigan Consumer Sentiment
    'USEPUINDXD': 'daily',      # Economic Policy Uncertainty Index
    
    # Housing (monthly - leading indicator)
    'MORTGAGE30US': 'weekly',   # 30-Year Mortgage Rate
    'HOUST': 'monthly',         # Housing Starts
    
    # Industrial Production
    'INDPRO': 'monthly',        # Industrial Production Index
    'TCU': 'monthly',           # Capacity Utilization
}

# =============================================================================
# BENCHMARK COMPARISON CONFIGURATION
# =============================================================================
# Audit benchmark settings (used by src/audit.py)
AUDIT_BENCHMARK_TICKER = 'SPY'  # S&P 500 proxy for relative-performance tracking
AUDIT_FORWARD_DAYS = 5          # Must align with 1-week performance labeling in audit
AUDIT_STALE_CLOSE_DAYS = 45     # Auto-close OPEN recommendations older than this if unresolved
AUDIT_BLOCK_ON_STALE_OPEN_ROWS = True
AUDIT_MAX_STALE_OPEN_ROWS = 0
AUDIT_MIN_DIRECTIONAL_SAMPLES = 200
AUDIT_MIN_BUY_SAMPLES = 30
AUDIT_MIN_BENCHMARK_DATES = 20
AUDIT_MIN_CONF_BUCKET_SAMPLES = 10
PUBLIC_REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "public")

# Optional daily notifications (used by src/notifier.py)
ENABLE_DAILY_NOTIFICATIONS = False
ENABLE_WEEKLY_NOTIFICATIONS = os.environ.get("ENABLE_WEEKLY_NOTIFICATIONS", "1").strip().lower() in {"1", "true", "yes", "on"}
NOTIFICATION_WEBHOOK_URL = os.environ.get("MARKET_NOTIFICATION_WEBHOOK_URL", "").strip()
NOTIFICATION_NTFY_TOPIC = os.environ.get("MARKET_NOTIFICATION_NTFY_TOPIC", "").strip()
NOTIFICATION_TIMEOUT_SECONDS = 10

# Dashboard scheduling defaults
DASHBOARD_AUDIT_DEFAULT_TIME = os.environ.get("DASHBOARD_AUDIT_DEFAULT_TIME", "18:30").strip() or "18:30"
DASHBOARD_AUDIT_DEFAULT_DAYS = [0, 1, 2, 3, 4]  # Mon-Fri (market days)
DASHBOARD_WEEKLY_SUMMARY_DEFAULT_TIME = os.environ.get("DASHBOARD_WEEKLY_SUMMARY_DEFAULT_TIME", "18:45").strip() or "18:45"
try:
    DASHBOARD_WEEKLY_SUMMARY_DEFAULT_DAY = int(os.environ.get("DASHBOARD_WEEKLY_SUMMARY_DEFAULT_DAY", "4"))  # Fri
except Exception:
    DASHBOARD_WEEKLY_SUMMARY_DEFAULT_DAY = 4
if DASHBOARD_WEEKLY_SUMMARY_DEFAULT_DAY < 0 or DASHBOARD_WEEKLY_SUMMARY_DEFAULT_DAY > 6:
    DASHBOARD_WEEKLY_SUMMARY_DEFAULT_DAY = 4

# Blend of indices to compare strategy performance against
# Keys are tickers, values are weights (must sum to 1.0)
BENCHMARK_BLEND = {
    'SPY': 0.40,   # S&P 500 (US Large Cap)
    'VTI': 0.35,   # Vanguard Total US Market
    'VXUS': 0.25,  # Vanguard Total International
}
