import json
import os
import sys
from typing import Dict, Optional, Tuple

# Import from centralized config FIRST (critical for thread control on ARM)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (N_JOBS, RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF,
                    GBM_N_ESTIMATORS, GBM_MAX_DEPTH, GBM_MIN_SAMPLES_LEAF, CV_SPLITS,
                    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, XGB_SUBSAMPLE, XGB_COLSAMPLE,
                    ENSEMBLE_WEIGHTS, PROJECT_ROOT, MODEL_DIR, MODEL_PATH, FEEDBACK_PATH,
                    ENABLE_XGBOOST, SCANNER_WORKERS, ML_FEATURE_CONTRACT,
                    HORIZONS, HORIZON_TARGETS, MODEL_PATH_5D, MODEL_PATH_10D, MODEL_PATH_30D,
                    USE_ADVANCED_DIVIDEND_FEATURES, MODEL_THRESHOLDS_FILE,
                    DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD,
                    RECENCY_WEIGHTING_ENABLED, RECENCY_HALF_LIFE_DAYS, RECENCY_MIN_WEIGHT,
                    QUAL_FEATURES_FILE, QUAL_MATURITY_MAP, QUAL_CYCLICAL_MAP, QUAL_MOAT_MAP, QUAL_DEBT_MAP,
                    QUAL_SECTOR_COLUMNS, QUAL_INDUSTRY_COLUMNS)


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import datetime
import pandas as pd
import numpy as np

from storage import load_price_data
from signals import generate_signals

# Try to import XGBoost (may not be installed or may be disabled in config)
HAS_XGBOOST = False
if ENABLE_XGBOOST:
    try:
        from xgboost import XGBClassifier
        HAS_XGBOOST = True
        print("XGBoost enabled and loaded.")
    except ImportError:
        print("Warning: XGBoost enabled but not installed. Using RF+GBM only.")
else:
    print("XGBoost disabled in config. Using RF+GBM ensemble only.")

class MarketModel:
    """Enhanced ensemble model combining RF, GBM, and optionally XGBoost."""
    _cached_model = None

    def __init__(self):
        self.rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, 
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            class_weight={0: 1.0, 1: 1.5},  # Asymmetric Loss: Penalize false positives
            random_state=42, n_jobs=N_JOBS
        )
        self.gbm = GradientBoostingClassifier(
            n_estimators=GBM_N_ESTIMATORS, max_depth=GBM_MAX_DEPTH, 
            min_samples_leaf=GBM_MIN_SAMPLES_LEAF,
            random_state=42
        )
        
        # Base estimators
        estimators = [('rf', self.rf)]
        weights = [1.0]

        if HAS_XGBOOST:
            # FAST MODE: Use XGBoost + RF (Drop slow Sklearn GBM)
            self.xgb = XGBClassifier(
                n_estimators=XGB_N_ESTIMATORS,
                max_depth=XGB_MAX_DEPTH,
                learning_rate=XGB_LEARNING_RATE,
                subsample=XGB_SUBSAMPLE,
                colsample_bytree=XGB_COLSAMPLE,
                eval_metric='logloss',
                n_jobs=N_JOBS,
                random_state=42
            )
            estimators.append(('xgb', self.xgb))
            weights = [1.0, 1.5]  # RF=1.0, XGB=1.5
            print("Ensemble: RF + XGBoost (Optimized for Speed)")
        else:
            # FALLBACK MODE: Use RF + GBM (Slow but works without XGB)
            estimators.append(('gbm', self.gbm))
            weights = [1.0, 1.0]
            print("Ensemble: RF + Sklearn GBM (Fallback - Slower)")
        
        self.model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights
        )
        
    def train(self, X, y, tracker=None, status_file=None, horizon_info=None, work_units_done: float = None,
              save_path: str = None, sample_weight: Optional[np.ndarray] = None):
        import sys
        # Clear cache before training to ensure we save the new one
        MarketModel._cached_model = None
        
        print(f"Starting CV training on {len(X)} samples with {len(X.columns)} features...")
        sys.stdout.flush()
        
        embargo = 0
        if horizon_info and isinstance(horizon_info, dict):
            try:
                embargo = int(horizon_info.get('current_horizon') or 0)
            except Exception:
                embargo = 0
        tscv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=embargo)
        scores = []
        last_val_proba = None
        last_val_y = None
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            if sample_weight.shape[0] != len(X):
                print("Warning: sample_weight length mismatch; disabling weights.", flush=True)
                sample_weight = None
        for fold_num, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            if tracker and status_file:
                # Data prep is often fast, CV is heavy.
                # Assume data prep was 50% (reached here).
                fold_progress = 0.5 + (0.5 * (fold_num - 1) / CV_SPLITS)
                # Calculate cumulative work done in this fold
                # Each fold processes len(train_idx) samples
                current_work = (work_units_done or 0) + len(train_idx)
                tracker.write_status(fold_progress, f"CV Training (Fold {fold_num}/{CV_SPLITS})", 
                                     status_file, horizon_info=horizon_info, work_units_done=current_work)

            print(f"  CV Fold {fold_num}/{CV_SPLITS}: Training on {len(train_idx)} samples...")
            sys.stdout.flush()
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            if sample_weight is not None:
                sw_train = sample_weight[train_idx]
                self.model.fit(X_train, y_train, sample_weight=sw_train)
            else:
                self.model.fit(X_train, y_train)
            score = self.model.score(X_val, y_val)
            scores.append(score)
            print(f"  CV Fold {fold_num}/{CV_SPLITS}: Done. Accuracy: {score:.3f}")
            sys.stdout.flush()

            if fold_num == CV_SPLITS:
                try:
                    last_val_proba = self.model.predict_proba(X_val)[:, 1]
                    last_val_y = y_val.values
                except Exception as e:
                    print(f"  Warning: Could not compute validation probabilities: {e}", flush=True)
        
        print(f"CV scores: {[f'{s:.3f}' for s in scores]}, Mean: {np.mean(scores):.3f}")
        sys.stdout.flush()

        buy_thr = DEFAULT_BUY_THRESHOLD
        sell_thr = DEFAULT_SELL_THRESHOLD
        calib_metrics = None
        if last_val_proba is not None and last_val_y is not None:
            buy_thr, sell_thr, calib_metrics = _calibrate_thresholds(last_val_y, last_val_proba)
            if horizon_info and isinstance(horizon_info, dict):
                horizon = horizon_info.get('current_horizon')
                if horizon:
                    save_model_thresholds(int(horizon), buy_thr, sell_thr, calib_metrics)
            print(f"Calibrated thresholds: BUY>={buy_thr:.2f}, SELL<={sell_thr:.2f}")
            sys.stdout.flush()
        
        print(f"Final fit on all {len(X)} samples...")
        sys.stdout.flush()
        
        if tracker and status_file:
            current_work = (work_units_done or 0) + len(X) # Final fit uses all X
            tracker.write_status(0.95, "Finalizing Model (Fitting Full Dataset)", status_file, 
                                 horizon_info=horizon_info, work_units_done=current_work)
            
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        
        # Save to specified path (or legacy MODEL_PATH for backward compat)
        actual_save_path = save_path or MODEL_PATH
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, actual_save_path)
        print(f"Ensemble model saved to {actual_save_path}")
        sys.stdout.flush()
        
        # Update cache
        MarketModel._cached_model = self.model
        
        
        # Log feature importance
        self._log_feature_importance(X.columns.tolist())
        
        print("\n=== TRAINING COMPLETE ===")
        sys.stdout.flush()
        
    def _log_feature_importance(self, feature_names):
        """Logs aggregated feature importances for analysis."""
        try:
            importances = []
            
            # RF Importance
            rf_model = self.model.named_estimators_.get('rf')
            if rf_model and hasattr(rf_model, 'feature_importances_'):
                importances.append(rf_model.feature_importances_)
            
            # XGB Importance (if available)
            if 'xgb' in self.model.named_estimators_:
                xgb_model = self.model.named_estimators_.get('xgb')
                if xgb_model and hasattr(xgb_model, 'feature_importances_'):
                    importances.append(xgb_model.feature_importances_)
            
            if importances:
                # Average across models
                avg_importance = np.mean(importances, axis=0)
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': avg_importance
                }).sort_values('importance', ascending=False)
                
                importance_path = os.path.join(MODEL_DIR, 'feature_importance.csv')
                importance_df.to_csv(importance_path, index=False)
                print(f"Feature importance saved. Top 5: {importance_df.head()['feature'].tolist()}")
        except Exception as e:
            print(f"Could not log feature importance: {e}")
        
    def predict_proba(self, X):
        if MarketModel._cached_model is None:
            if not os.path.exists(MODEL_PATH):
                return np.ones(len(X)) * 0.5
            MarketModel._cached_model = joblib.load(MODEL_PATH)
            
        proba = MarketModel._cached_model.predict_proba(X)[:, 1]
        return np.clip(proba, 0, 1)


def get_dividend_features(ticker: str) -> dict:
    """
    Fetches enhanced dividend features for ML training.
    Includes: Yield, Safety Score, Yield Trap Detection.
    """
    try:
        import yfinance as yf
        from datetime import datetime
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        div_yield = info.get('dividendYield', 0) or 0
        div_rate = info.get('dividendRate', 0) or 0
        ex_div_date = info.get('exDividendDate', None)
        payout_ratio = info.get('payoutRatio', 0) or 0
        
        # Calculate days to ex-div
        days_to_ex_div = None
        if ex_div_date:
            ex_date = datetime.fromtimestamp(ex_div_date)
            days_to_ex_div = (ex_date - datetime.now()).days
        
        # === DIVIDEND SAFETY SCORE (0-1) ===
        # Lower payout = safer. Penalize > 80%.
        payout_score = max(0, 1 - (payout_ratio / 0.8)) if payout_ratio <= 1.5 else 0
        # Debt/Equity penalty (if available)
        debt_equity = info.get('debtToEquity', 0) or 0
        debt_penalty = max(0, 1 - (debt_equity / 200)) if debt_equity else 1.0
        dividend_safety_score = min(1.0, (payout_score * 0.7 + debt_penalty * 0.3))
        
        # === YIELD TRAP DETECTION ===
        # High yield + big price drop = potential trap
        fifty_two_week_low = info.get('fiftyTwoWeekLow', 0) or 0
        fifty_two_week_high = info.get('fiftyTwoWeekHigh', 1) or 1
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 50))
        price_pct_from_high = (current_price - fifty_two_week_high) / fifty_two_week_high if fifty_two_week_high else 0
        
        is_yield_trap = (div_yield > 0.08 and price_pct_from_high < -0.30)  # >8% yield AND >30% off high
        
        # === DIVIDEND GROWTH (Approximation) ===
        # Try to get 5yr avg growth from trailingAnnualDividendRate vs fiveYearAvgDividendYield
        # This is a rough proxy since yfinance doesn't give full history easily
        five_yr_avg_yield = info.get('fiveYearAvgDividendYield', 0) or 0
        dividend_growth_5y = 0.0
        if five_yr_avg_yield > 0 and div_yield > 0:
            # Rough growth = current yield / avg yield - 1 (inverted logic, assumes stable payout)
            dividend_growth_5y = max(-0.5, min(0.5, (div_yield / five_yr_avg_yield) - 1))
        
        return {
            'dividend_yield': div_yield,
            'dividend_rate': div_rate,
            'days_to_ex_div': days_to_ex_div,
            'payout_ratio': payout_ratio,
            'is_dividend': div_yield > 0.01,  # >1% yield
            'dividend_safety_score': dividend_safety_score,
            'is_yield_trap': is_yield_trap,
            'dividend_growth_5y': dividend_growth_5y,
        }
    except:
        return {'dividend_yield': 0, 'dividend_rate': 0, 'days_to_ex_div': None, 
                'payout_ratio': 0, 'is_dividend': False, 'dividend_safety_score': 0.5,
                'is_yield_trap': False, 'dividend_growth_5y': 0}


def _load_thresholds_file() -> dict:
    try:
        if os.path.exists(MODEL_THRESHOLDS_FILE):
            with open(MODEL_THRESHOLDS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"[thresholds] Warning: Could not load {MODEL_THRESHOLDS_FILE}: {e}", flush=True)
    return {}


def _write_thresholds_file(data: dict) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    temp_path = MODEL_THRESHOLDS_FILE + '.tmp'
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(temp_path, MODEL_THRESHOLDS_FILE)


def _f1_score_binary(y_true: np.ndarray, y_pred_positive: np.ndarray, positive_label: int) -> float:
    """Compute F1 for a given positive label using boolean predictions."""
    pos_mask = y_true == positive_label
    pred_pos = y_pred_positive.astype(bool)
    tp = np.sum(pos_mask & pred_pos)
    fp = np.sum(~pos_mask & pred_pos)
    fn = np.sum(pos_mask & ~pred_pos)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _calibrate_thresholds(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
    """Calibrate BUY/SELL thresholds from validation probabilities."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    if y_true.size == 0 or y_true.size != proba.size:
        return DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD, {'reason': 'empty'}

    pos_count = int(np.sum(y_true == 1))
    neg_count = int(np.sum(y_true == 0))
    if pos_count < 10 or neg_count < 10:
        return DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD, {'reason': 'insufficient_class_samples'}

    best_buy = DEFAULT_BUY_THRESHOLD
    best_buy_f1 = -1.0
    for thr in np.linspace(0.50, 0.90, 41):
        preds = proba >= thr
        f1 = _f1_score_binary(y_true, preds, positive_label=1)
        if f1 > best_buy_f1:
            best_buy_f1 = f1
            best_buy = float(thr)

    best_sell = DEFAULT_SELL_THRESHOLD
    best_sell_f1 = -1.0
    for thr in np.linspace(0.10, 0.50, 41):
        preds = proba <= thr
        f1 = _f1_score_binary(y_true, preds, positive_label=0)
        if f1 > best_sell_f1:
            best_sell_f1 = f1
            best_sell = float(thr)

    if best_sell >= best_buy:
        return DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD, {'reason': 'overlap'}

    return best_buy, best_sell, {
        'pos_count': float(pos_count),
        'neg_count': float(neg_count),
        'buy_f1': float(best_buy_f1),
        'sell_f1': float(best_sell_f1)
    }


def save_model_thresholds(horizon: int, buy_threshold: float, sell_threshold: float,
                          metrics: Optional[Dict[str, float]] = None) -> None:
    data = _load_thresholds_file()
    data[str(horizon)] = {
        'buy': round(float(buy_threshold), 4),
        'sell': round(float(sell_threshold), 4),
        'updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'metrics': metrics or {}
    }
    data['last_updated'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    _write_thresholds_file(data)


def _compute_recency_weights(dates: pd.Series) -> np.ndarray:
    """Compute exponential decay weights by recency."""
    if dates is None or len(dates) == 0:
        return np.ones(0)
    try:
        dt = pd.to_datetime(dates)
    except Exception:
        return np.ones(len(dates))
    if RECENCY_HALF_LIFE_DAYS <= 0:
        return np.ones(len(dt))
    max_date = dt.max()
    age_days = (max_date - dt).dt.days.astype(float).clip(lower=0)
    weights = 0.5 ** (age_days / float(RECENCY_HALF_LIFE_DAYS))
    return np.clip(weights, RECENCY_MIN_WEIGHT, 1.0).to_numpy()


# =============================================================================
# QUALITATIVE FEATURE LOADING & EXTRACTION
# =============================================================================

_qual_cache = None

def load_qual_features() -> dict:
    """
    Load qualitative features from cache file.
    Returns dict of ticker -> qual_data. Empty dict on any failure (graceful).
    """
    global _qual_cache
    if _qual_cache is not None:
        return _qual_cache
    
    try:
        with open(QUAL_FEATURES_FILE, 'r') as f:
            import json
            data = json.load(f)
        _qual_cache = data.get('tickers', {})
    except Exception as e:
        print(f"[qual_features] Warning: Could not load qual features: {e}")
        _qual_cache = {}
    
    return _qual_cache


def get_qual_data(ticker: str) -> dict:
    """Get qual data for a ticker. Returns None if not found (uses defaults)."""
    cache = load_qual_features()
    return cache.get(ticker.upper())


def extract_qual_features(qual_data: dict = None) -> dict:
    """
    Extract qualitative features with proper encoding.
    Returns dict of feature_name -> value.
    
    Features:
    - qual_is_etf: 1 if ETF, 0 if stock
    - qual_confidence: mean confidence across all fields (lets model weight trust)
    - qual_maturity: ordinal 0-3
    - qual_cyclical: ordinal 0-2
    - qual_moat: ordinal 0-2
    - qual_debt: ordinal 0-2
    - qual_sector_*: one-hot encoded sector (13 columns)
    - qual_industry_*: one-hot encoded industry (33 columns)
    """
    features = {}
    
    # Initialize all columns to defaults
    features['qual_is_etf'] = 0
    features['qual_confidence'] = 0.0
    features['qual_maturity'] = 2  # mature default
    features['qual_cyclical'] = 1  # mixed default
    features['qual_moat'] = 1      # narrow default
    features['qual_debt'] = 1      # medium default
    
    # Initialize all one-hot columns to 0
    for col in QUAL_SECTOR_COLUMNS:
        features[col] = 0
    for col in QUAL_INDUSTRY_COLUMNS:
        features[col] = 0
    
    # Set unknown flags when no data
    features['qual_sector_unknown'] = 1
    features['qual_industry_unknown'] = 1
    
    if not qual_data:
        return features
    
    # We have data!
    features['qual_is_etf'] = 1 if qual_data.get('is_etf') else 0
    
    # Confidence handling
    conf = qual_data.get('confidence', {})
    MIN_CONFIDENCE = 0.5
    
    # Calculate mean confidence (let model weight trust)
    conf_values = [v for v in conf.values() if isinstance(v, (int, float))]
    features['qual_confidence'] = sum(conf_values) / len(conf_values) if conf_values else 0.0
    
    # Sector one-hot encoding
    if conf.get('sector', 0) >= MIN_CONFIDENCE:
        sector = qual_data.get('sector', 'Unknown')
        col_name = 'qual_sector_' + sector.lower().replace(' ', '_').replace('&', 'and')
        if col_name in QUAL_SECTOR_COLUMNS:
            features['qual_sector_unknown'] = 0  # Clear unknown flag
            features[col_name] = 1
    
    # Industry one-hot encoding
    if conf.get('industry', 0) >= MIN_CONFIDENCE:
        industry = qual_data.get('industry', 'Unknown')
        col_name = 'qual_industry_' + industry.lower().replace(' ', '_').replace('&', 'and')
        if col_name in QUAL_INDUSTRY_COLUMNS:
            features['qual_industry_unknown'] = 0  # Clear unknown flag
            features[col_name] = 1
    
    # Ordinal classifications (only if confidence >= threshold)
    classifications = qual_data.get('classifications', {})
    
    if conf.get('business_maturity', 0) >= MIN_CONFIDENCE:
        val = classifications.get('business_maturity', 'mature')
        features['qual_maturity'] = QUAL_MATURITY_MAP.get(val, 2)
    
    if conf.get('cyclical_exposure', 0) >= MIN_CONFIDENCE:
        val = classifications.get('cyclical_exposure', 'mixed')
        features['qual_cyclical'] = QUAL_CYCLICAL_MAP.get(val, 1)
    
    if conf.get('moat_strength', 0) >= MIN_CONFIDENCE:
        val = classifications.get('moat_strength', 'narrow')
        features['qual_moat'] = QUAL_MOAT_MAP.get(val, 1)
    
    if conf.get('debt_risk', 0) >= MIN_CONFIDENCE:
        val = classifications.get('debt_risk', 'medium')
        features['qual_debt'] = QUAL_DEBT_MAP.get(val, 1)
    
    return features


def extract_ml_features(df: pd.DataFrame, dividend_info: dict = None, macro_data: dict = None, 
                        qual_data: dict = None, horizon: int = 10, for_inference: bool = False) -> pd.DataFrame:
    """
    Extracts ML features including technicals, volume, volatility, dividend, and qualitative info.
    
    Args:
        df: Price DataFrame with OHLCV and technical indicators
        dividend_info: Optional dividend data
        macro_data: Optional macro indicators
        qual_data: Optional qualitative/industry data from gemini-cli
        horizon: Prediction horizon in trading days (5, 10, or 30)
    
    Dividend yield is treated as a POSITIVE signal because:
    - Total return = price appreciation + dividends
    - High-yield stocks provide income even if price is flat
    - Dividend growth often indicates company quality
    """
    df = df.copy()

    
    # === MOMENTUM FEATURES ===
    df['returns_1d'] = df['close'].pct_change(1)
    df['returns_5d'] = df['close'].pct_change(5)
    df['returns_10d'] = df['close'].pct_change(10)
    df['returns_20d'] = df['close'].pct_change(20)
    
    # === VOLATILITY FEATURES ===
    df['volatility_20d'] = df['returns_1d'].rolling(20).std()
    vol_median = df['volatility_20d'].rolling(60).median()
    df['volatility_regime'] = (df['volatility_20d'] > vol_median).astype(int)
    
    # ATR for volatility-adjusted signals
    if 'high' in df.columns and 'low' in df.columns:
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift())
        df['low_close'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr_14'] = df['tr'].rolling(14).mean()
        df['atr_ratio'] = df['atr_14'] / df['close']  # Normalized ATR
    else:
        df['atr_14'] = 0  # Default when high/low not available
        df['atr_ratio'] = 0
    
    # === TREND FEATURES ===
    df['dist_sma_50'] = (df['close'] / df['sma_50']) - 1
    df['dist_sma_200'] = (df['close'] / df['sma_200']) - 1
    df['sma_50_slope'] = df['sma_50'].pct_change(5)
    
    # Golden/Death cross proximity
    df['sma_cross_dist'] = (df['sma_50'] - df['sma_200']) / df['sma_200']
    
    # 52-week high distance (stocks near highs tend to continue)
    df['price_vs_52w_high'] = df['close'] / df['close'].rolling(252).max() - 1
    
    # === RSI FEATURES ===
    df['rsi_slope'] = df['rsi'].diff(3)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    
    # === MACD FEATURES ===
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_hist_slope'] = df['macd_histogram'].diff(3)
    else:
        df['macd_histogram'] = 0
        df['macd_hist_slope'] = 0
    
    # === BOLLINGER BANDS ===
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - bb_mid) / (2 * bb_std + 1e-8)  # -1 to +1 range
    
    # === VOLUME FEATURES ===
    if 'volume' in df.columns:
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_zscore'] = (df['volume'] - df['volume_sma_20']) / (df['volume'].rolling(20).std() + 1e-8)
        df['volume_trend'] = df['volume'].rolling(5).mean() / (df['volume'].rolling(20).mean() + 1e-8)
    else:
        df['volume_zscore'] = 0
        df['volume_trend'] = 1
    
    # === DIVIDEND FEATURES (Enhanced) ===
    if dividend_info and not USE_ADVANCED_DIVIDEND_FEATURES:
        dividend_info = {
            'dividend_yield': dividend_info.get('dividend_yield', 0),
            'dividend_rate': dividend_info.get('dividend_rate', 0),
            'is_dividend': dividend_info.get('is_dividend', False),
        }
    # Includes: Yield, Safety Score, Yield Trap Warning, Growth
    if dividend_info:
        df['dividend_yield'] = dividend_info.get('dividend_yield', 0)
        days = dividend_info.get('days_to_ex_div')
        # Normalize days to ex-div (0-1 scale, closer = higher value)
        df['days_to_ex_div_norm'] = max(0, 1 - (days / 30)) if days and days > 0 else 0
        df['is_dividend_stock'] = 1 if dividend_info.get('is_dividend') else 0
        # Payout ratio quality signal (too high = risky, too low = no commitment)
        payout = dividend_info.get('payout_ratio', 0)
        df['payout_quality'] = 1 if 0.3 <= payout <= 0.7 else 0.5
        # NEW: Safety, Trap, Growth
        df['dividend_safety_score'] = dividend_info.get('dividend_safety_score', 0.5)
        df['is_yield_trap'] = 1 if dividend_info.get('is_yield_trap') else 0
        df['dividend_growth_5y'] = dividend_info.get('dividend_growth_5y', 0)
    else:
        df['dividend_yield'] = 0
        df['days_to_ex_div_norm'] = 0
        df['is_dividend_stock'] = 0
        df['payout_quality'] = 0.5
        df['dividend_safety_score'] = 0.5
        df['is_yield_trap'] = 0
        df['dividend_growth_5y'] = 0
    
    
    # === MACRO FEATURES (Centralized Fetching) ===
    # Use passed macro_data if available, otherwise fallback to defaults (prevents per-row fetching spam)
    macro = macro_data or {}
    
    df['macro_vix'] = macro.get('macro_VIXCLS', 20)  # Default to 20 if missing
    df['macro_yield_curve'] = macro.get('macro_T10Y2Y', 0)
    df['macro_hy_spread'] = macro.get('macro_BAMLH0A0HYM2', 4)
    df['macro_stress'] = macro.get('macro_STLFSI4', 0)
    df['macro_fed_assets_chg'] = macro.get('macro_WALCL_chg1m', 0) / 1e12  # Normalize to trillions
    
    # === TIME FEATURES ===
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'])
        df['day_of_week'] = dates.dt.dayofweek
        df['month'] = dates.dt.month
        df['week_of_year'] = dates.dt.isocalendar().week.astype(int)
        df['day_of_year'] = dates.dt.dayofyear
        df['is_month_start'] = dates.dt.is_month_start.astype(int)
        df['is_month_end'] = dates.dt.is_month_end.astype(int)
    else:
        # Fallbacks for dateless frame (should allow basic prediction)
        df['day_of_week'] = 2
        df['month'] = 6
        df['week_of_year'] = 26
        df['day_of_year'] = 180
        df['is_month_start'] = 0
        df['is_month_end'] = 0

    # === DYNAMIC INTERACTION TERMS ===
    # Cross Top Technicals with Top Macro
    # Technicals: RSI (Oversold), Volatility, Dist SMA 200 (Trend)
    # Macro: VIX (Fear), Yield Curve (Recession)
    
    # VIX Interactions (Fear Regime)
    # High VIX * Low RSI = Panic Selling Opportunity?
    df['inter_vix_rsi'] = df['macro_vix'] * df['rsi']
    df['inter_vix_sma200'] = df['macro_vix'] * df['dist_sma_200']
    
    # Yield Curve Interactions (Growth/Recession Regime)
    # Inverted curve * Low SMA = Recession Dump?
    df['inter_yield_sma200'] = df['macro_yield_curve'] * df['dist_sma_200']
    df['inter_yield_rsi'] = df['macro_yield_curve'] * df['rsi']

    # === QUALITATIVE FEATURES ===
    # Static industry/sector features from gemini-cli (graceful defaults if missing)
    qual_features = extract_qual_features(qual_data)
    for col, val in qual_features.items():
        df[col] = val

    
    if not for_inference:
        # === TRIPLE BARRIER META-LABELING (Realistic Trading Target) ===
        # A trade is only a WIN if it hits the Profit Target BEFORE hitting the Stop Loss or Time Limit.
        # Profit Target = Dynamic based on Volatility (ATR) AND horizon. Longer horizons = higher targets.
        # Stop Loss = Scaled by horizon (2% for 10d, 1.5% for 5d, 4% for 30d).
        # Time Limit = Horizon days.
        
        # Get horizon-specific target from config
        base_target = HORIZON_TARGETS.get(horizon, 0.03)  # Default 3% for 10d
        
        # 1. Define Barriers
        # Dynamic Profit Target: Minimum is horizon-based, scaled by 0.5 * ATR%
        target_pct = np.maximum(base_target, 0.5 * df['atr_ratio'])
        
        # Stop loss scales with horizon (longer = more room), adaptive to volatility
        min_stop = {5: 0.015, 10: 0.02, 30: 0.04}.get(horizon, 0.02)
        stop_pct = np.maximum(min_stop, 1.5 * df['atr_ratio'])
        
        # 2. Calculate Barrier Prices for each row
        profit_price = df['close'] * (1 + target_pct)
        stop_price = df['close'] * (1 - stop_pct)
        
        # 3. Look Forward N Days based on horizon
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)
        future_low_min = df['low'].shift(-1).rolling(window=indexer).min()
        future_high_max = df['high'].shift(-1).rolling(window=indexer).max()
        
        # 4. Determine Outcome
        # WIN if: (Max > Profit) AND (Min > Stop)
        # We defensively assume that if Min < Stop, the Stop was hit FIRST or concurrently.
        # This filters out "volatile shakes" that would stop us out before the rally.
        
        # Adjust target for dividends (Total Return view)
        if dividend_info and dividend_info.get('is_dividend'):
            # Dividends reduce the price appreciation needed
            div_yield_q = dividend_info.get('dividend_yield', 0) / 4
            profit_price = df['close'] * (1 + np.maximum(0.01, target_pct - div_yield_q))

        is_win = (future_high_max >= profit_price) & (future_low_min > stop_price)
        
        df['meta_label'] = is_win.astype(int)
        
        # Drop rows with NaN (from rolling calcs at start) AND last N rows (no future data for horizon)
        df_clean = df.dropna().iloc[:-horizon]
    else:
        # Inference mode: keep latest rows, no future-label trimming.
        df_clean = df.dropna()

    # === CONTRACT ENFORCEMENT & ALIGNMENT ===
    # Ensure correct columns and ORDER for ML inference
    # Fill missing with 0, drop extra
    for col in ML_FEATURE_CONTRACT:
        if col not in df_clean.columns:
            df_clean[col] = 0.0
            
    if for_inference:
        return df_clean[ML_FEATURE_CONTRACT]
    return df_clean[ML_FEATURE_CONTRACT + ['meta_label']] 


# FEATURE_COLS: "Source of Truth" for the entire system
# Imported from config's contract to prevent drift
FEATURE_COLS = ML_FEATURE_CONTRACT




def _process_ticker_for_training(ticker, ticker_div_map, macro_data, qual_cache, horizon=10):
    """Helper function to process a single ticker for training data."""
    try:
        # Check if we have enough data first (fast check)
        # Note: We rely on load_price_data which uses a read-only connection
        # to avoid locking issues in threads.
        df = load_price_data(ticker)
        if len(df) < 250: 
            return None
            
        # Use pre-fetched dividend info
        div_info = ticker_div_map.get(ticker) or {}
        
        # Get qual data (if available)
        qual_data = qual_cache.get(ticker.upper()) if qual_cache else None
        
        # Generate signals and features with horizon-specific target
        df = generate_signals(df, ticker=ticker, dividend_info=div_info)
        features = extract_ml_features(df, dividend_info=div_info, macro_data=macro_data, 
                                       qual_data=qual_data, horizon=horizon)
        
        # Filter valid numeric features
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'date', 'ticker', 'meta_label', 'signal']
        feature_cols = [c for c in features.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(features[c])]
        
        if 'meta_label' in features.columns and len(feature_cols) > 10:
            result = features[feature_cols + ['meta_label']].copy()
            if 'date' in df.columns:
                try:
                    result['date'] = pd.to_datetime(df.loc[features.index, 'date']).values
                except Exception:
                    result['date'] = df.loc[features.index, 'date'].values
            return result
            
    except Exception as e:
        print(f"    Worker Error for {ticker}: {e}", flush=True)
        return None
        
    return None


def prepare_training_data(tickers, tracker=None, status_file=None, horizon=10, horizon_info=None,
                          work_units_done: float = None, return_weights: bool = False):
    """
    Loads data, generates signals, and prepares X/y for training.
    Uses parallel processing to maximize CPU utilization on M1 Ultra.
    
    Args:
        tickers: List of tickers to process
        tracker: ETA tracker for status updates
        status_file: Path to status file
        horizon: Prediction horizon in days (5, 10, or 30)
    """
    print(f"  Pre-fetching dividend info for {len(tickers)} tickers (horizon={horizon}d)...", flush=True)
    if tracker and status_file:
        tracker.write_status(0.1, f"Pre-fetching Dividends ({horizon}d)", status_file, 
                             horizon_info=horizon_info, work_units_done=work_units_done)

    # Keep feature extraction within the global ML core budget (N_JOBS) to avoid
    # nested oversubscription while still allowing aggressive ticker parallelism.
    n_workers = max(1, SCANNER_WORKERS * 4)
    n_workers = min(n_workers, max(1, N_JOBS))
    if n_workers < 4 and N_JOBS >= 4:
        n_workers = 4

    # Pre-fetch dividends
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        div_infos = list(executor.map(get_dividend_features, tickers))
    
    ticker_div_map = dict(zip(tickers, div_infos))
    
    all_features = []
    total_tickers = len(tickers)
    
    # Load macro data once
    from macro_loader import get_macro_features
    try:
        macro_data = get_macro_features()
    except Exception as e:
        print(f"  Warning: Failed to load macro data ({e}). Using empty defaults.", flush=True)
        macro_data = {}
    
    # Load qualitative features cache (graceful if missing)
    qual_cache = load_qual_features()
    if qual_cache:
        print(f"  Loaded qual features for {len(qual_cache)} tickers.", flush=True)
    
    print(f"  Extracting features ({horizon}d horizon, Parallel: {n_workers} workers)...", flush=True)
    
    # Parallel Feature Extraction
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks with horizon parameter and qual_cache
        future_to_ticker = {
            executor.submit(_process_ticker_for_training, ticker, ticker_div_map, macro_data, qual_cache, horizon): ticker 
            for ticker in tickers
        }
        print(f"  Submitted {len(future_to_ticker)} tasks to executor.", flush=True)
        
        processed_count = 0
        for future in as_completed(future_to_ticker):
            processed_count += 1
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                if result is not None:
                    all_features.append(result)
            except Exception as e:
                import traceback
                print(f"    Error processing {ticker} (result): {e}", flush=True)
                traceback.print_exc()
                
            # Update status more frequently (every 10 tickers)
            if processed_count % 10 == 0 or processed_count == total_tickers:
                 print(f"    Processed {processed_count}/{total_tickers} tickers...", flush=True)
                 if tracker and status_file:
                     # Features extraction phase is 0.15 to 0.50 of total progress
                     prog = 0.15 + (0.35 * processed_count / total_tickers)
                     # Small increment during feature extraction
                     current_work = (work_units_done or 0) + (processed_count * 10) 
                     tracker.write_status(prog, f"Extracting Features ({processed_count}/{total_tickers})", 
                                          status_file, horizon_info=horizon_info, work_units_done=current_work)

        print(f"  Finished extracting features. Total processed: {processed_count}/{total_tickers}. Valid datasets: {len(all_features)}", flush=True)

    if not all_features:
        if return_weights:
            return pd.DataFrame(), pd.Series(), np.array([])
        return pd.DataFrame(), pd.Series()
    
    print(f"  Concatenating {len(all_features)} feature sets...", flush=True)
    try:
        data = pd.concat(all_features)
    except ValueError:
        # Handle case where all results were empty/None logic fell through
        if return_weights:
            return pd.DataFrame(), pd.Series(), np.array([])
        return pd.DataFrame(), pd.Series()
    
    # Ensure time ordering for TimeSeriesSplit (use date if available)
    sample_weights = None
    if 'date' in data.columns:
        data = data.sort_values('date').reset_index(drop=True)
        if return_weights:
            if RECENCY_WEIGHTING_ENABLED:
                sample_weights = _compute_recency_weights(data['date'])
            else:
                sample_weights = np.ones(len(data))
        data = data.drop(columns=['date'])
    elif return_weights:
        sample_weights = np.ones(len(data))

    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in data.columns:
            data[col] = 0
            
    # CRITICAL: Numerical Stability Cleanup
    # Replace infinity with NaN, then fill all NaNs with 0
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Clip extreme values to prevent 'value too large for float32' errors
    # Most technical indicators and macro ratios shouldn't be in the millions.
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].clip(lower=-1e6, upper=1e6)
    
    X_out = data[FEATURE_COLS]
    y_out = data['meta_label']
    if return_weights:
        if sample_weights is None or len(sample_weights) != len(X_out):
            sample_weights = np.ones(len(X_out))
        return X_out, y_out, sample_weights
    return X_out, y_out


def train_market_model():
    """
    Trains ensemble models for all prediction horizons (5d, 10d, 30d).
    Each horizon gets its own model file for specialized predictions.
    """
    # Robust import for storage (handle running from root or src)
    try:
        from storage import get_existing_tickers
    except ImportError:
        # Add src to sys.path
        src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from storage import get_existing_tickers

    from eta_tracker import ETATracker
    from config import PROJECT_ROOT
    
    status_file = os.path.join(PROJECT_ROOT, 'TRAIN_STATUS.md')
    tracker = ETATracker("ml_training")
    
    tickers = get_existing_tickers()
    print(f"Training Multi-Horizon Ensemble on {len(tickers)} tickers...", flush=True)
    print(f"Horizons: {HORIZONS} days", flush=True)
    print(f"Models: RF + GBM" + (" + XGBoost" if HAS_XGBOOST else ""), flush=True)
    
    # Start ETA tracking
    tracker.start_run({"tickers": len(tickers), "horizons": len(HORIZONS)})
    
    # Estimate Total Work Units (Samples x horizons) for Overall ETA
    # Assuming avg 1000 samples per ticker x 3 horizons
    total_est_samples = len(tickers) * 1000 * len(HORIZONS)
    tracker.set_work_units(total_est_samples)
    
    # Create a persistent session tracker for overall timing
    session_tracker = ETATracker("ml_session")
    session_tracker.start_run({"total_horizons": len(HORIZONS)})
    session_tracker.set_work_units(total_est_samples)
    tracker.session_tracker = session_tracker
    
    tracker.write_status(0.01, "Initializing Multi-Horizon Training", status_file)
    
    # Model paths for each horizon
    horizon_model_paths = {
        5: MODEL_PATH_5D,
        10: MODEL_PATH_10D,  # Consistent naming for 10d model
        30: MODEL_PATH_30D,
    }
    
    total_horizons = len(HORIZONS)
    
    for horizon_idx, horizon in enumerate(HORIZONS):
        horizon_progress_base = horizon_idx / total_horizons
        horizon_progress_step = 1.0 / total_horizons
        
        # Horizon info for status display
        horizon_info = {
            'current_horizon': horizon,
            'horizon_idx': horizon_idx,
            'total_horizons': total_horizons
        }
        
        print(f"\n{'='*50}", flush=True)
        print(f"Training {horizon}-Day Model ({horizon_idx + 1}/{total_horizons})...", flush=True)
        print(f"{'='*50}", flush=True)
        
        tracker.write_status(
            horizon_progress_base + 0.01, 
            f"Training {horizon}d Model ({horizon_idx + 1}/{total_horizons})", 
            status_file,
            horizon_info=horizon_info
        )
        
        # Prepare training data for this horizon
        # Cumulative work calculation for overall ETA
        cv_samples_per_horizon = len(tickers) * 1000 # Rough estimate
        cum_work_done = horizon_idx * cv_samples_per_horizon

        X, y, sample_weights = prepare_training_data(
            tickers, 
            tracker=tracker, 
            status_file=status_file,
            horizon=horizon,
            horizon_info=horizon_info,
            work_units_done=cum_work_done,
            return_weights=True
        )
        
        if X.empty:
            print(f"  WARNING: No data for {horizon}d model. Skipping.", flush=True)
            continue
        
        print(f"  Data: {X.shape}, Classes: {y.value_counts().to_dict()}", flush=True)
        
        # Train model for this horizon
        model = MarketModel()
        
        # Temporarily override MODEL_PATH for this horizon
        original_model_path = MODEL_PATH
        model_save_path = horizon_model_paths.get(horizon, MODEL_PATH)
        
        # Train the model and save directly to horizon-specific path (no overwriting)
        model.train(
            X,
            y,
            tracker=tracker,
            status_file=status_file,
            horizon_info=horizon_info,
            save_path=model_save_path,
            sample_weight=sample_weights
        )
        
        tracker.write_status(
            horizon_progress_base + horizon_progress_step * 0.95,
            f"Completed {horizon}d Model",
            status_file,
            horizon_info=horizon_info
        )
    
    # End tracking
    tracker.write_status(1.0, "All Horizons Complete", status_file)
    tracker.end_run()
    
    print(f"\n{'='*50}", flush=True)
    print(f"MULTI-HORIZON TRAINING COMPLETE", flush=True)
    print(f"Models saved: {list(horizon_model_paths.values())}", flush=True)
    print(f"{'='*50}", flush=True)


def get_signal_confidence(ticker_row: pd.Series) -> float:
    """Get ML confidence for a single ticker's latest data (10d model for backward compatibility)."""
    available_cols = [c for c in FEATURE_COLS if c in ticker_row.index]
    if len(available_cols) < 5:
        return 0.5
    
    # Fill missing columns with 0
    X_dict = {c: ticker_row.get(c, 0) for c in FEATURE_COLS}
    X = pd.DataFrame([X_dict])[FEATURE_COLS]
    
    model = MarketModel()
    return float(model.predict_proba(X)[0])


def get_multi_horizon_confidence(ticker_row: pd.Series) -> dict:
    """
    Get ML confidence for all 3 prediction horizons.
    
    Returns:
        dict with keys: 'conf_5d', 'conf_10d', 'conf_30d'
    """
    available_cols = [c for c in FEATURE_COLS if c in ticker_row.index]
    if len(available_cols) < 5:
        return {'conf_5d': 0.5, 'conf_10d': 0.5, 'conf_30d': 0.5}
    
    # Fill missing columns with 0
    X_dict = {c: ticker_row.get(c, 0) for c in FEATURE_COLS}
    X = pd.DataFrame([X_dict])[FEATURE_COLS]
    
    results = {}
    
    # Model paths for each horizon
    horizon_paths = {
        'conf_5d': MODEL_PATH_5D,
        'conf_10d': MODEL_PATH_10D,  # Consistent 10d model
        'conf_30d': MODEL_PATH_30D,
    }
    
    for key, path in horizon_paths.items():
        try:
            if os.path.exists(path):
                model = joblib.load(path)
                proba = model.predict_proba(X)[:, 1]
                results[key] = float(np.clip(proba[0], 0, 1))
            else:
                # Model not yet trained
                results[key] = 0.5
        except Exception as e:
            print(f"  Warning: Could not load {key} model: {e}", flush=True)
            results[key] = 0.5
    
    return results


def learn_from_performance():
    """
    Analyzes past recommendations to adjust model confidence.
    Learns from successful and failed predictions.
    """
    from storage import get_connection
    con = get_connection()
    
    # Get closed recommendations with performance data
    df = con.execute("""
        SELECT * FROM recommendation_history 
        WHERE status = 'CLOSED' AND perf_1w IS NOT NULL
    """).df()
    con.close()
    
    if len(df) < 20:
        print("Not enough closed recommendations for feedback learning.")
        return
    
    # Calculate success rate for high-confidence predictions
    high_conf = df[df['confidence'] > 0.6]
    if len(high_conf) > 0:
        buy_success = high_conf[(high_conf['signal_type'] == 'BUY') & (high_conf['perf_1w'] > 0)]
        accuracy = len(buy_success) / len(high_conf[high_conf['signal_type'] == 'BUY']) if len(high_conf[high_conf['signal_type'] == 'BUY']) > 0 else 0.5
        
        # Boost or penalize future confidence based on accuracy
        accuracy_boost = 0.8 + (accuracy * 0.4)  # Range: 0.8 to 1.2
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump({'accuracy_boost': accuracy_boost, 'sample_size': len(high_conf)}, FEEDBACK_PATH)
        print(f"Feedback learning complete. Accuracy: {accuracy:.1%}, Boost factor: {accuracy_boost:.2f}")
