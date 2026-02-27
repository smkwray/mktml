import duckdb
import pandas as pd
import os
import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'market_data.duckdb')


def _normalize_dividend_yield(value) -> float:
    """Normalize dividend yield to decimal format."""
    try:
        yld = float(value or 0.0)
    except Exception:
        return 0.0
    if not np.isfinite(yld) or yld <= 0:
        return 0.0
    for _ in range(3):
        if yld <= 1.0:
            break
        yld = yld / 100.0
    return max(0.0, yld)


def _get_universe_denyset() -> set:
    """Load configured ticker denylist for write-time filtering."""
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from config import UNIVERSE_DENYLIST
    except Exception:
        return set()
    return {str(t).upper().strip() for t in (UNIVERSE_DENYLIST or []) if str(t).strip()}

def get_connection(read_only=False):
    """Returns a connection to the DuckDB database."""
    return duckdb.connect(DB_PATH, read_only=read_only)

def initialize_db():
    """Creates the necessary tables if they don't exist."""
    con = get_connection()
    
    # Create price_history table (legacy alias 'prices' for consistency if needed, but keeping 'price_history' as per original schema to avoid breaking changes)
    con.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            ticker VARCHAR,
            PRIMARY KEY (ticker, date)
        )
    """)
    # Note: 'price_history' is legacy alias, ensuring we use 'prices' or keep consistent. 
    # Current code uses 'price_history' in queries but creates it here.
    
    con.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals (
            ticker VARCHAR PRIMARY KEY,
            dividend_yield DOUBLE,
            pe_ratio DOUBLE,
            market_cap DOUBLE,
            sector VARCHAR,
            industry VARCHAR,
            updated_at TIMESTAMP
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS ticker_metadata (
            ticker VARCHAR PRIMARY KEY,
            sector VARCHAR,
            industry VARCHAR,
            country VARCHAR,
            market_cap DOUBLE,
            updated_at TIMESTAMP
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_history (
            date DATE,
            ticker VARCHAR,
            signal_type VARCHAR,
            price_at_rec DOUBLE,
            rsi DOUBLE,
            sma_50 DOUBLE,
            sma_200 DOUBLE,
            confidence DOUBLE,
            conf_5d DOUBLE,
            conf_30d DOUBLE,
            perf_1w DOUBLE,
            perf_1m DOUBLE,
            status VARCHAR DEFAULT 'OPEN',
            reason VARCHAR,
            atr_ratio DOUBLE,
            dividend_yield DOUBLE,
            is_safe_asset BOOLEAN,
            returns_30d DOUBLE,
            tradable BOOLEAN,
            tradability_reason VARCHAR,
            avg_volume_20d DOUBLE,
            avg_dollar_volume_20d DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS model_predictions (
            asof_date DATE,
            ticker VARCHAR,
            horizon INTEGER,
            proba_raw DOUBLE,
            proba_cal DOUBLE,
            score DOUBLE,
            signal_type VARCHAR,
            tradable BOOLEAN,
            tradability_reason VARCHAR,
            filters_triggered VARCHAR,
            model_version_hash VARCHAR,
            data_snapshot_hash VARCHAR,
            created_at TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (asof_date, ticker, horizon, model_version_hash)
        )
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_recommendation_history_date_ticker
        ON recommendation_history(date, ticker)
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_recommendation_history_status_date
        ON recommendation_history(status, date)
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_model_predictions_asof_horizon
        ON model_predictions(asof_date, horizon)
    """)
    
    # Schema Migration: Add 'reason' column if it doesn't exist
    try:
        con.execute("ALTER TABLE recommendation_history ADD COLUMN reason VARCHAR")
        print("Schema updated: Added 'reason' column to recommendation_history")
    except:
        pass # Column likely already exists

    # Schema Migration: Add multi-horizon confidence columns
    try:
        con.execute("ALTER TABLE recommendation_history ADD COLUMN conf_5d DOUBLE")
        print("Schema updated: Added 'conf_5d' column to recommendation_history")
    except:
        pass
    try:
        con.execute("ALTER TABLE recommendation_history ADD COLUMN conf_30d DOUBLE")
        print("Schema updated: Added 'conf_30d' column to recommendation_history")
    except:
        pass
        
    # Schema Migration: Add Safe Asset columns
    try:
        con.execute("ALTER TABLE recommendation_history ADD COLUMN atr_ratio DOUBLE")
        con.execute("ALTER TABLE recommendation_history ADD COLUMN dividend_yield DOUBLE")
        con.execute("ALTER TABLE recommendation_history ADD COLUMN is_safe_asset BOOLEAN")
        print("Schema updated: Added Safe Asset columns")
    except:
        pass # Columns likely exist
    try:
        con.execute("ALTER TABLE recommendation_history ADD COLUMN returns_30d DOUBLE")
        print("Schema updated: Added returns_30d column")
    except:
        pass # Column likely exists
    try:
        con.execute("ALTER TABLE recommendation_history ADD COLUMN tradable BOOLEAN")
        print("Schema updated: Added tradable column")
    except:
        pass
    try:
        con.execute("ALTER TABLE recommendation_history ADD COLUMN tradability_reason VARCHAR")
        print("Schema updated: Added tradability_reason column")
    except:
        pass
    try:
        con.execute("ALTER TABLE recommendation_history ADD COLUMN avg_volume_20d DOUBLE")
        print("Schema updated: Added avg_volume_20d column")
    except:
        pass
    try:
        con.execute("ALTER TABLE recommendation_history ADD COLUMN avg_dollar_volume_20d DOUBLE")
        print("Schema updated: Added avg_dollar_volume_20d column")
    except:
        pass

    con.close()
    print(f"Database initialized at {DB_PATH}")

def save_fundamentals(ticker: str, data: dict):
    """Saves or updates ticker fundamentals."""
    if not data:
        return
    dividend_yield = _normalize_dividend_yield(data.get('dividend_yield', 0))
    con = get_connection()
    try:
        # Use INSERT OR REPLACE
        con.execute("""
            INSERT OR REPLACE INTO fundamentals (ticker, dividend_yield, pe_ratio, market_cap, sector, industry, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, current_timestamp)
        """, [
            ticker, 
            dividend_yield,
            data.get('pe_ratio', 0), 
            data.get('market_cap', 0),
            data.get('sector'),
            data.get('industry')
        ])
    except Exception as e:
        print(f"Error saving fundamentals for {ticker}: {e}")
    finally:
        con.close()

def get_fundamentals(ticker: str) -> dict:
    """Retrieves fundamentals for a ticker."""
    con = get_connection(read_only=True)
    try:
        res = con.execute("SELECT * FROM fundamentals WHERE ticker = ?", [ticker]).df()
        if not res.empty:
            out = res.iloc[0].to_dict()
            out['dividend_yield'] = _normalize_dividend_yield(out.get('dividend_yield', 0))
            return out
        return {}
    except Exception:
        return {}
    finally:
        con.close()

def save_metadata(metadata: dict):
    """Saves or updates ticker metadata."""
    if not metadata:
        return
    con = get_connection()
    try:
        # Use INSERT OR REPLACE
        con.execute("""
            INSERT OR REPLACE INTO ticker_metadata (ticker, sector, industry, country, market_cap, updated_at)
            VALUES (?, ?, ?, ?, ?, current_timestamp)
        """, [metadata['ticker'], metadata.get('sector'), metadata.get('industry'), metadata.get('country'), metadata.get('market_cap')])
    except Exception as e:
        print(f"Error saving metadata for {metadata.get('ticker')}: {e}")
    finally:
        con.close()

def get_metadata(ticker: str) -> dict:
    """Retrieves metadata for a ticker."""
    con = get_connection(read_only=True)
    try:
        res = con.execute("SELECT * FROM ticker_metadata WHERE ticker = ?", [ticker]).df()
        if not res.empty:
            return res.iloc[0].to_dict()
        return {}
    except Exception:
        return {}
    finally:
        con.close()

def save_recommendations(recs: list):
    """Saves a list of recommendations to history."""
    if not recs:
        return
    con = get_connection()
    df = pd.DataFrame(recs).copy()
    
    # Ensure new columns exist with defaults if not provided
    if 'conf_5d' not in df.columns:
        df['conf_5d'] = 0.5
    if 'conf_30d' not in df.columns:
        df['conf_30d'] = 0.5
    if 'reason' not in df.columns:
        df['reason'] = None
    if 'atr_ratio' not in df.columns:
        df['atr_ratio'] = 0.0
    if 'dividend_yield' not in df.columns:
        df['dividend_yield'] = 0.0
    if 'is_safe_asset' not in df.columns:
        df['is_safe_asset'] = False
    if 'returns_30d' not in df.columns:
        df['returns_30d'] = None
    if 'tradable' not in df.columns:
        df['tradable'] = True
    if 'tradability_reason' not in df.columns:
        df['tradability_reason'] = 'unknown'
    if 'avg_volume_20d' not in df.columns:
        df['avg_volume_20d'] = 0.0
    if 'avg_dollar_volume_20d' not in df.columns:
        df['avg_dollar_volume_20d'] = 0.0
    if 'status' not in df.columns:
        df['status'] = 'OPEN'
    else:
        df['status'] = df['status'].fillna('OPEN')
    if 'confidence' not in df.columns:
        df['confidence'] = 0.5

    # Normalize keys and deduplicate: one recommendation per ticker/date.
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    denyset = _get_universe_denyset()
    if denyset:
        before = len(df)
        df = df[~df['ticker'].isin(denyset)].copy()
        removed = before - len(df)
        if removed > 0:
            print(f"Dropped {removed} denylisted recommendation(s): {sorted(denyset)}")
    df = df.dropna(subset=['date', 'ticker'])
    if df.empty:
        con.close()
        return
    df = df.sort_values(['date', 'ticker', 'confidence'], kind='mergesort')
    df = df.drop_duplicates(subset=['date', 'ticker'], keep='last')
    df_keys = df[['date', 'ticker']].drop_duplicates()
    
    try:
        replaced = con.execute("""
            SELECT COUNT(*) FROM recommendation_history
            WHERE EXISTS (
                SELECT 1 FROM df_keys k
                WHERE recommendation_history.date = k.date
                  AND upper(recommendation_history.ticker) = upper(k.ticker)
            )
        """).fetchone()[0]
        con.execute("""
            DELETE FROM recommendation_history
            WHERE EXISTS (
                SELECT 1 FROM df_keys k
                WHERE recommendation_history.date = k.date
                  AND upper(recommendation_history.ticker) = upper(k.ticker)
            )
        """)
        con.execute("""
            INSERT INTO recommendation_history 
            (date, ticker, signal_type, price_at_rec, rsi, sma_50, sma_200, confidence, conf_5d, conf_30d, status, reason, atr_ratio, dividend_yield, is_safe_asset, returns_30d, tradable, tradability_reason, avg_volume_20d, avg_dollar_volume_20d)
            SELECT date, ticker, signal_type, price_at_rec, rsi, sma_50, sma_200, confidence, conf_5d, conf_30d, status, reason, atr_ratio, dividend_yield, is_safe_asset, returns_30d, tradable, tradability_reason, avg_volume_20d, avg_dollar_volume_20d
            FROM df
        """)
        print(f"Saved {len(df)} recommendations ({replaced} replaced for same ticker/date).")
    except Exception as e:
        print(f"Error saving recommendations: {e}")
    finally:
        con.close()

def save_price_data(df: pd.DataFrame):
    """Saves price data to the database."""
    if df.empty:
        return

    con = get_connection()
    expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
    df_to_save = df[expected_cols].copy()
    # Normalize to DATE to avoid timestamp-vs-date duplicate key collisions.
    df_to_save['date'] = pd.to_datetime(df_to_save['date'], errors='coerce').dt.date
    df_to_save = df_to_save.dropna(subset=['date', 'ticker'])
    df_to_save = df_to_save.drop_duplicates(subset=['ticker', 'date'], keep='last')
    if df_to_save.empty:
        con.close()
        return
    
    try:
        # Upsert by primary key so late backfills/corrections are not silently dropped.
        con.execute("INSERT OR REPLACE INTO price_history SELECT * FROM df_to_save")
        print(f"Saved {len(df_to_save)} rows to database (upserted by ticker/date).")
    except Exception as e:
        print(f"Error saving data: {e}")
    finally:
        con.close()


def save_model_predictions(rows: list):
    """Upsert replay/model prediction rows."""
    if not rows:
        return

    con = get_connection()
    df = pd.DataFrame(rows).copy()
    if df.empty:
        con.close()
        return

    defaults = {
        'proba_raw': 0.5,
        'proba_cal': 0.5,
        'score': 0.5,
        'signal_type': 'HOLD',
        'tradable': True,
        'tradability_reason': 'unknown',
        'filters_triggered': '',
        'model_version_hash': '',
        'data_snapshot_hash': '',
    }
    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = df[col].fillna(default_val)

    # Canonicalize floating-point probabilities/scores to avoid tiny IEEE drift
    # causing false replay determinism mismatches across identical runs.
    for col in ('proba_raw', 'proba_cal', 'score'):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').fillna(0.5).astype(float).to_numpy()
            vals = np.clip(vals, 0.0, 1.0)
            df[col] = np.round(vals, 12)

    if 'ticker' not in df.columns or 'asof_date' not in df.columns or 'horizon' not in df.columns:
        con.close()
        raise ValueError("save_model_predictions requires ticker, asof_date, and horizon columns")

    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    df['asof_date'] = pd.to_datetime(df['asof_date'], errors='coerce').dt.date
    df['horizon'] = pd.to_numeric(df['horizon'], errors='coerce').fillna(0).astype(int)
    df = df.dropna(subset=['ticker', 'asof_date'])
    if df.empty:
        con.close()
        return

    key_cols = ['asof_date', 'ticker', 'horizon', 'model_version_hash']
    df = df.sort_values(key_cols, kind='mergesort').drop_duplicates(subset=key_cols, keep='last')
    df_keys = df[key_cols].drop_duplicates()

    try:
        replaced = con.execute("""
            SELECT COUNT(*) FROM model_predictions
            WHERE EXISTS (
                SELECT 1 FROM df_keys k
                WHERE model_predictions.asof_date = k.asof_date
                  AND upper(model_predictions.ticker) = upper(k.ticker)
                  AND model_predictions.horizon = k.horizon
                  AND model_predictions.model_version_hash = k.model_version_hash
            )
        """).fetchone()[0]
        con.execute("""
            DELETE FROM model_predictions
            WHERE EXISTS (
                SELECT 1 FROM df_keys k
                WHERE model_predictions.asof_date = k.asof_date
                  AND upper(model_predictions.ticker) = upper(k.ticker)
                  AND model_predictions.horizon = k.horizon
                  AND model_predictions.model_version_hash = k.model_version_hash
            )
        """)
        con.execute("""
            INSERT INTO model_predictions
            (asof_date, ticker, horizon, proba_raw, proba_cal, score, signal_type, tradable, tradability_reason, filters_triggered, model_version_hash, data_snapshot_hash)
            SELECT asof_date, ticker, horizon, proba_raw, proba_cal, score, signal_type, tradable, tradability_reason, filters_triggered, model_version_hash, data_snapshot_hash
            FROM df
        """)
        print(f"Saved {len(df)} model prediction rows ({replaced} replaced).")
    except Exception as e:
        print(f"Error saving model predictions: {e}")
    finally:
        con.close()

def load_price_data(ticker: str) -> pd.DataFrame:
    """Loads price data for a specific ticker (Read Only)."""
    con = get_connection(read_only=True)
    try:
        query = "SELECT * FROM price_history WHERE ticker = ? ORDER BY date"
        df = con.execute(query, [ticker]).df()
        return df
    finally:
        con.close()

def get_existing_tickers():
    """Returns a list of tickers already in the database (Read Only)."""
    con = get_connection(read_only=True)
    try:
        tickers = con.execute("SELECT DISTINCT ticker FROM price_history").fetchall()
        return [t[0] for t in tickers]
    finally:
        con.close()
