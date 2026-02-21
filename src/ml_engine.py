import json
import hashlib
import os
import sys
from typing import Dict, Optional, Tuple, List, Any

# Import from centralized config FIRST (critical for thread control on ARM)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (N_JOBS, RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF,
                    GBM_N_ESTIMATORS, GBM_MAX_DEPTH, GBM_MIN_SAMPLES_LEAF, CV_SPLITS,
                    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, XGB_SUBSAMPLE, XGB_COLSAMPLE,
                    ENSEMBLE_WEIGHTS, PROJECT_ROOT, MODEL_DIR, MODEL_PATH, FEEDBACK_PATH,
                    ENABLE_XGBOOST, SCANNER_WORKERS, ML_FEATURE_CONTRACT,
                    CALIBRATION_ARTIFACT_DIR, CALIBRATION_BINS, CALIBRATION_MIN_SAMPLES,
                    PROB_CALIBRATION_FILE_TEMPLATE,
                    HORIZONS, HORIZON_TARGETS, MODEL_PATH_5D, MODEL_PATH_10D, MODEL_PATH_30D,
                    ASSET_BUCKET_MODELING_ENABLED, ASSET_BUCKETS, ASSET_BUCKET_MIN_TICKERS, ASSET_BUCKET_MIN_SAMPLES,
                    BUCKET_MODEL_DIR, WALK_FORWARD_ENABLED, WALK_FORWARD_SPLITS, WALK_FORWARD_TOP_QUANTILE,
                    WALK_FORWARD_MIN_TEST_SAMPLES, WALK_FORWARD_GROUP_BY_DATE,
                    WALK_FORWARD_TOP_K_FRAC, WALK_FORWARD_TOP_K_MIN_SAMPLES,
                    MODEL_VALIDATION_FILE,
                    USE_ADVANCED_DIVIDEND_FEATURES, MODEL_THRESHOLDS_FILE,
                    DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD,
                    RECENCY_WEIGHTING_ENABLED, RECENCY_HALF_LIFE_DAYS, RECENCY_MIN_WEIGHT,
                    QUAL_FEATURES_FILE, QUAL_MATURITY_MAP, QUAL_CYCLICAL_MAP, QUAL_MOAT_MAP, QUAL_DEBT_MAP,
                    QUAL_SECTOR_COLUMNS, QUAL_INDUSTRY_COLUMNS)


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import datetime
import pandas as pd
import numpy as np

from storage import load_price_data
from signals import generate_signals

# Reduce import-time noise when scripts need machine-readable stdout (JSON-only).
QUIET_IMPORTS = str(os.environ.get('MARKET_QUIET_IMPORTS', '0')).strip().lower() in {'1', 'true', 'yes', 'on'}

# Try to import XGBoost (may not be installed or may be disabled in config)
HAS_XGBOOST = False
if ENABLE_XGBOOST:
    try:
        from xgboost import XGBClassifier
        HAS_XGBOOST = True
        if not QUIET_IMPORTS:
            print("XGBoost enabled and loaded.")
    except Exception as e:
        if not QUIET_IMPORTS:
            print(f"Warning: XGBoost unavailable ({e}). Using RF+GBM only.")
else:
    if not QUIET_IMPORTS:
        print("XGBoost disabled in config. Using RF+GBM ensemble only.")


class StableEnsembleModel:
    """Serializable ensemble wrapper with optional XGBoost stable-model loading."""

    def __init__(
        self,
        rf_model: Any = None,
        gbm_model: Any = None,
        xgb_model_path: str = '',
        rf_weight: float = 1.0,
        gbm_weight: float = 1.0,
        xgb_weight: float = 1.5,
    ):
        self.rf_model = rf_model
        self.gbm_model = gbm_model
        self.xgb_model_path = str(xgb_model_path or '')
        self.rf_weight = float(rf_weight)
        self.gbm_weight = float(gbm_weight)
        self.xgb_weight = float(xgb_weight)
        self._xgb_model = None
        self._xgb_load_failed = False

    def _load_xgb_model(self):
        if not self.xgb_model_path or self._xgb_load_failed:
            return None
        if self._xgb_model is not None:
            return self._xgb_model
        if not os.path.exists(self.xgb_model_path):
            self._xgb_load_failed = True
            return None
        try:
            from xgboost import XGBClassifier

            model = XGBClassifier()
            model.load_model(self.xgb_model_path)
            self._xgb_model = model
            return self._xgb_model
        except Exception as e:
            print(f"[stable-ensemble] Warning: could not load xgboost model {self.xgb_model_path}: {e}", flush=True)
            self._xgb_load_failed = True
            return None

    def predict_proba(self, X):
        horizon_proba: List[np.ndarray] = []
        weights: List[float] = []

        if self.rf_model is not None and hasattr(self.rf_model, 'predict_proba'):
            horizon_proba.append(np.asarray(self.rf_model.predict_proba(X)[:, 1], dtype=float))
            weights.append(max(0.0, float(self.rf_weight)))
        if self.gbm_model is not None and hasattr(self.gbm_model, 'predict_proba'):
            horizon_proba.append(np.asarray(self.gbm_model.predict_proba(X)[:, 1], dtype=float))
            weights.append(max(0.0, float(self.gbm_weight)))

        xgb_model = self._load_xgb_model()
        if xgb_model is not None and hasattr(xgb_model, 'predict_proba'):
            horizon_proba.append(np.asarray(xgb_model.predict_proba(X)[:, 1], dtype=float))
            weights.append(max(0.0, float(self.xgb_weight)))

        if not horizon_proba:
            p1 = np.ones(len(X), dtype=float) * 0.5
        else:
            stacked = np.vstack(horizon_proba)
            w = np.asarray(weights, dtype=float)
            if np.sum(w) <= 0:
                p1 = np.mean(stacked, axis=0)
            else:
                p1 = np.average(stacked, axis=0, weights=w)
        p1 = np.clip(np.asarray(p1, dtype=float), 0.0, 1.0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def _build_stable_ensemble_from_voting(model: Any, save_path: str) -> Optional[StableEnsembleModel]:
    """Convert a fitted VotingClassifier into StableEnsembleModel and export XGB as .ubj."""
    named_estimators = getattr(model, 'named_estimators_', None)
    if not isinstance(named_estimators, dict):
        return None

    weight_map: Dict[str, float] = {}
    est_defs = getattr(model, 'estimators', None) or []
    raw_weights = list(getattr(model, 'weights', []) or [])
    if raw_weights and len(raw_weights) == len(est_defs):
        for (name, _), weight in zip(est_defs, raw_weights):
            try:
                weight_map[str(name)] = float(weight)
            except Exception:
                weight_map[str(name)] = 1.0
    else:
        for name, _ in est_defs:
            weight_map[str(name)] = 1.0

    rf_model = named_estimators.get('rf')
    gbm_model = named_estimators.get('gbm')
    xgb_model = named_estimators.get('xgb')

    xgb_model_path = ''
    if xgb_model is not None and hasattr(xgb_model, 'save_model'):
        base, _ = os.path.splitext(save_path)
        xgb_model_path = f"{base}.ubj"
        try:
            xgb_model.save_model(xgb_model_path)
        except Exception as e:
            print(f"[stable-ensemble] Warning: failed to save XGBoost stable model {xgb_model_path}: {e}", flush=True)
            xgb_model_path = ''

    return StableEnsembleModel(
        rf_model=rf_model,
        gbm_model=gbm_model,
        xgb_model_path=xgb_model_path,
        rf_weight=weight_map.get('rf', 1.0),
        gbm_weight=weight_map.get('gbm', 1.0),
        xgb_weight=weight_map.get('xgb', 1.5),
    )


def migrate_model_pickle_to_stable(path: str) -> Dict[str, Any]:
    """Migrate a model pickle to StableEnsembleModel and export XGBoost stable model when present."""
    abs_path = os.path.abspath(path)
    result: Dict[str, Any] = {
        'path': abs_path,
        'status': 'unknown',
        'xgb_model_path': '',
    }
    if not os.path.exists(abs_path):
        result['status'] = 'missing'
        result['error'] = 'file_not_found'
        return result

    try:
        model = joblib.load(abs_path)
    except Exception as exc:
        result['status'] = 'load_failed'
        result['error'] = f'{exc.__class__.__name__}: {exc}'
        return result

    if isinstance(model, StableEnsembleModel):
        result['status'] = 'already_stable'
        result['xgb_model_path'] = str(getattr(model, 'xgb_model_path', '') or '')
        return result

    stable_model = _build_stable_ensemble_from_voting(model, abs_path)
    if stable_model is None:
        # Non-voting models are re-saved with current runtime for sklearn version consistency.
        joblib.dump(model, abs_path)
        result['status'] = 'repickled_unmodified'
        return result

    joblib.dump(stable_model, abs_path)
    result['status'] = 'migrated'
    result['xgb_model_path'] = stable_model.xgb_model_path
    return result

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
              save_path: str = None, sample_weight: Optional[np.ndarray] = None,
              save_thresholds_for_horizon: bool = True,
              sample_intervals: Optional[pd.DataFrame] = None):
        import sys
        # Clear cache before training to ensure we save the new one
        MarketModel._cached_model = None
        actual_save_path = save_path or MODEL_PATH
        
        print(f"Starting CV training on {len(X)} samples with {len(X.columns)} features...")
        sys.stdout.flush()
        
        embargo = 0
        if horizon_info and isinstance(horizon_info, dict):
            try:
                embargo = int(horizon_info.get('current_horizon') or 0)
            except Exception:
                embargo = 0
        split_mode = "row_time_gap"
        split_pairs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        if sample_intervals is not None:
            split_pairs, purge_stats, split_mode = _build_purged_walk_forward_splits(
                sample_intervals=sample_intervals,
                n_splits=CV_SPLITS,
                gap_days=max(1, embargo),
                prefer_date_blocks=WALK_FORWARD_GROUP_BY_DATE,
            )
            if split_pairs:
                print(
                    f"Using {split_mode} CV ({len(split_pairs)} folds, "
                    f"purged={int(purge_stats.get('purged_total', 0))}, "
                    f"max_overlap_after={int(purge_stats.get('max_overlap_after_purge', 0))})",
                    flush=True,
                )

        if split_pairs is None:
            tscv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=embargo)
            split_pairs = list(tscv.split(X))

        total_folds = len(split_pairs)
        if total_folds == 0:
            raise ValueError("No valid CV folds generated for model training.")

        scores = []
        oof_val_proba: List[np.ndarray] = []
        oof_val_y: List[np.ndarray] = []
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            if sample_weight.shape[0] != len(X):
                print("Warning: sample_weight length mismatch; disabling weights.", flush=True)
                sample_weight = None
        for fold_num, (train_idx, val_idx) in enumerate(split_pairs, 1):
            if tracker and status_file:
                # Data prep is often fast, CV is heavy.
                # Assume data prep was 50% (reached here).
                fold_progress = 0.5 + (0.5 * (fold_num - 1) / max(1, total_folds))
                # Calculate cumulative work done in this fold
                # Each fold processes len(train_idx) samples
                current_work = (work_units_done or 0) + len(train_idx)
                tracker.write_status(fold_progress, f"CV Training (Fold {fold_num}/{total_folds})", 
                                     status_file, horizon_info=horizon_info, work_units_done=current_work)

            print(
                f"  CV Fold {fold_num}/{total_folds} [{split_mode}]: "
                f"Training on {len(train_idx)} samples...",
                flush=True,
            )
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
            print(f"  CV Fold {fold_num}/{total_folds}: Done. Accuracy: {score:.3f}")
            sys.stdout.flush()

            try:
                fold_val_proba = self.model.predict_proba(X_val)[:, 1]
                oof_val_proba.append(np.asarray(fold_val_proba, dtype=float))
                oof_val_y.append(np.asarray(y_val.values, dtype=int))
            except Exception as e:
                print(f"  Warning: Could not compute validation probabilities: {e}", flush=True)
        
        print(f"CV scores: {[f'{s:.3f}' for s in scores]}, Mean: {np.mean(scores):.3f}")
        sys.stdout.flush()

        buy_thr = DEFAULT_BUY_THRESHOLD
        sell_thr = DEFAULT_SELL_THRESHOLD
        calib_metrics = None
        if oof_val_proba and oof_val_y:
            all_val_proba = np.concatenate(oof_val_proba)
            all_val_y = np.concatenate(oof_val_y)
            buy_thr, sell_thr, calib_metrics = _calibrate_thresholds(all_val_y, all_val_proba)
            if save_thresholds_for_horizon and horizon_info and isinstance(horizon_info, dict):
                horizon = horizon_info.get('current_horizon')
                if horizon:
                    save_model_thresholds(int(horizon), buy_thr, sell_thr, calib_metrics)
                    try:
                        calibration_path = save_probability_calibration(
                            horizon=int(horizon),
                            raw_proba=all_val_proba,
                            y_true=all_val_y,
                            model_path=actual_save_path,
                            model_status=horizon_info,
                        )
                        print(f"Saved probability calibration artifact: {calibration_path}")
                    except Exception as e:
                        print(f"  Warning: Failed to save probability calibration for {horizon}d: {e}", flush=True)
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
        os.makedirs(MODEL_DIR, exist_ok=True)
        stable_model = _build_stable_ensemble_from_voting(self.model, actual_save_path)
        if stable_model is not None:
            joblib.dump(stable_model, actual_save_path)
            if stable_model.xgb_model_path:
                print(
                    f"Ensemble model saved to {actual_save_path} "
                    f"(xgboost exported: {stable_model.xgb_model_path})"
                )
            else:
                print(f"Ensemble model saved to {actual_save_path} (stable wrapper)")
        else:
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


def _sha256_file(path: str) -> str:
    """Return SHA256 for provenance checks; unreadable files return 'unreadable'."""
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


def _calibration_path_for_horizon(horizon: int) -> str:
    """Return deterministic path to the per-horizon calibration artifact."""
    return os.path.join(
        CALIBRATION_ARTIFACT_DIR,
        PROB_CALIBRATION_FILE_TEMPLATE.format(horizon=int(horizon)),
    )


def _load_probability_calibration(horizon: int) -> Optional[Dict[str, object]]:
    """Load cached probability calibration profile for a horizon."""
    if not isinstance(horizon, int):
        return None
    cached = _CALIBRATION_ARTIFACT_CACHE.get(int(horizon))
    if cached is not None:
        return cached

    path = _calibration_path_for_horizon(horizon)
    if not os.path.exists(path):
        return None

    try:
        with open(path, 'r') as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return None
        if int(payload.get('horizon', -1)) != int(horizon):
            return None
        if not isinstance(payload.get('bin_edges'), list) or not isinstance(payload.get('bin_calibrated'), list):
            return None
        _CALIBRATION_ARTIFACT_CACHE[int(horizon)] = payload
        return payload
    except Exception:
        return None


def _build_probability_calibration(
    y_true: np.ndarray,
    raw_proba: np.ndarray,
) -> Optional[Dict[str, object]]:
    """Build deterministic bin-based calibration profile."""
    y_arr = np.asarray(y_true).astype(float)
    raw_arr = np.asarray(raw_proba).astype(float)
    if y_arr.size == 0 or raw_arr.size != y_arr.size:
        return None

    valid = np.isfinite(y_arr) & np.isfinite(raw_arr)
    y_arr = y_arr[valid]
    raw_arr = np.clip(raw_arr[valid], 0.0, 1.0)

    sample_count = int(y_arr.size)
    if sample_count == 0:
        return None

    overall_rate = float(np.mean(y_arr))
    if sample_count < int(CALIBRATION_MIN_SAMPLES):
        return {
            'horizon': 0,
            'bin_count': int(CALIBRATION_BINS),
            'bin_edges': [0.0, 1.0],
            'bin_calibrated': [overall_rate],
            'num_samples': sample_count,
            'num_valid_samples': sample_count,
            'overall_rate': overall_rate,
            'min_raw': float(np.min(raw_arr)),
            'max_raw': float(np.max(raw_arr)),
            'method': 'fallback_global',
            'status': 'insufficient_samples',
        }

    n_bins = int(max(5, CALIBRATION_BINS))
    edges = np.linspace(0.0, 1.0, n_bins + 1).tolist()
    bin_idx = np.digitize(raw_arr, edges[1:-1], right=True)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    calibrated = np.full(n_bins, overall_rate, dtype=float)
    for b in range(n_bins):
        mask = bin_idx == b
        if np.any(mask):
            calibrated[b] = float(np.mean(y_arr[mask]))

    return {
        'horizon': 0,
        'bin_count': n_bins,
        'bin_edges': [float(v) for v in edges],
        'bin_calibrated': [float(v) for v in calibrated.tolist()],
        'num_samples': sample_count,
        'num_valid_samples': sample_count,
        'overall_rate': overall_rate,
        'min_raw': float(np.min(raw_arr)),
        'max_raw': float(np.max(raw_arr)),
        'method': 'fixed_bin_rates',
        'status': 'ok',
    }


def _apply_probability_calibration(raw_prob: float, horizon: int) -> float:
    """Map a raw probability to a calibrated value for a horizon."""
    try:
        raw_prob_value = float(raw_prob)
    except (TypeError, ValueError):
        return 0.5

    artifact = _load_probability_calibration(int(horizon))
    if artifact is None:
        return float(np.clip(raw_prob_value, 0.0, 1.0))

    edges = artifact.get('bin_edges')
    values = artifact.get('bin_calibrated')
    if not isinstance(edges, list) or not isinstance(values, list):
        return float(np.clip(raw_prob_value, 0.0, 1.0))
    if not edges or not values:
        return float(np.clip(raw_prob_value, 0.0, 1.0))

    try:
        edges_arr = np.array(edges, dtype=float)
        values_arr = np.array(values, dtype=float)
        if edges_arr.size < 2 or values_arr.size == 0:
            return float(np.clip(raw_prob_value, 0.0, 1.0))
        idx = int(
            np.digitize(
                float(np.clip(raw_prob_value, 0.0, 1.0)),
                edges_arr[1:-1],
                right=True,
            )
        )
        idx = int(np.clip(idx, 0, values_arr.size - 1))
        return float(np.clip(values_arr[idx], 0.0, 1.0))
    except Exception:
        return float(np.clip(raw_prob_value, 0.0, 1.0))


def save_probability_calibration(
    horizon: int,
    raw_proba: np.ndarray,
    y_true: np.ndarray,
    model_path: str = '',
    model_status: Optional[Dict[str, object]] = None,
) -> str:
    """Persist probability calibration artifact and return file path."""
    profile = _build_probability_calibration(y_true=y_true, raw_proba=raw_proba)
    if profile is None:
        raise ValueError("No data available for calibration artifact.")

    profile = dict(profile)
    profile['horizon'] = int(horizon)
    profile['status'] = profile.get('status', 'ok')
    profile['generated_at'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    profile['model_path'] = model_path
    if model_path:
        profile['model_sha256'] = _sha256_file(model_path)
    if model_status:
        profile['model_status'] = dict(model_status)

    os.makedirs(CALIBRATION_ARTIFACT_DIR, exist_ok=True)
    artifact_path = _calibration_path_for_horizon(int(horizon))
    tmp_path = artifact_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(profile, f, indent=2)
    os.replace(tmp_path, artifact_path)
    return artifact_path


def _get_threshold_diagnostics(horizon: int) -> Dict[str, Any]:
    """Return threshold diagnostics for validation/promotion reporting."""
    horizon_int = int(horizon)
    data = _load_thresholds_file()
    entry = data.get(str(horizon_int), {}) if isinstance(data, dict) else {}
    has_entry = isinstance(entry, dict) and ('buy' in entry or 'sell' in entry)

    if has_entry:
        try:
            buy = float(entry.get('buy', DEFAULT_BUY_THRESHOLD))
        except (TypeError, ValueError):
            buy = float(DEFAULT_BUY_THRESHOLD)
        try:
            sell = float(entry.get('sell', DEFAULT_SELL_THRESHOLD))
        except (TypeError, ValueError):
            sell = float(DEFAULT_SELL_THRESHOLD)
        source = 'calibrated_file'
        updated = entry.get('updated')
        metrics = entry.get('metrics') if isinstance(entry.get('metrics'), dict) else {}
    else:
        buy = float(DEFAULT_BUY_THRESHOLD)
        sell = float(DEFAULT_SELL_THRESHOLD)
        source = 'default_fallback'
        updated = None
        metrics = {}

    return {
        'horizon': horizon_int,
        'source': source,
        'buy': buy,
        'sell': sell,
        'updated': updated,
        'metrics': metrics,
    }


def _get_calibration_diagnostics(horizon: int, model_path: str = '') -> Dict[str, Any]:
    """Return probability calibration artifact diagnostics for reporting."""
    horizon_int = int(horizon)
    artifact_path = _calibration_path_for_horizon(horizon_int)
    artifact = _load_probability_calibration(horizon_int)
    diagnostics: Dict[str, Any] = {
        'horizon': horizon_int,
        'artifact_path': artifact_path,
        'artifact_exists': bool(artifact is not None),
        'status': 'missing',
        'method': None,
        'num_samples': 0,
        'num_valid_samples': 0,
        'bin_count': 0,
        'generated_at': None,
        'model_sha256': None,
        'model_sha_match': None,
    }
    if artifact is None:
        return diagnostics

    diagnostics['status'] = str(artifact.get('status', 'unknown'))
    diagnostics['method'] = artifact.get('method')
    diagnostics['num_samples'] = int(artifact.get('num_samples') or 0)
    diagnostics['num_valid_samples'] = int(artifact.get('num_valid_samples') or 0)
    diagnostics['bin_count'] = int(artifact.get('bin_count') or 0)
    diagnostics['generated_at'] = artifact.get('generated_at')
    diagnostics['model_sha256'] = artifact.get('model_sha256')

    if model_path and diagnostics['model_sha256']:
        try:
            diagnostics['model_sha_match'] = bool(_sha256_file(model_path) == diagnostics['model_sha256'])
        except Exception:
            diagnostics['model_sha_match'] = None
    return diagnostics


def _build_promotion_gate_status(
    wf_result: Dict[str, Any],
    threshold_diag: Dict[str, Any],
    calibration_diag: Dict[str, Any],
) -> Dict[str, Any]:
    """Build explicit promotion gate status for validation outputs."""
    bucket = str(wf_result.get('bucket', ''))
    if bucket != "GLOBAL":
        return {
            'applicable': False,
            'pass': None,
            'reason': 'global_only',
        }

    min_wf_samples = int(max(1, WALK_FORWARD_MIN_TEST_SAMPLES * max(2, WALK_FORWARD_SPLITS)))
    checks = {
        'walk_forward_available': not bool(wf_result.get('skipped', False)),
        'sample_count_ok': int(wf_result.get('samples') or 0) >= min_wf_samples,
        'thresholds_available': str(threshold_diag.get('source', '')) == 'calibrated_file',
        'calibration_artifact_present': bool(calibration_diag.get('artifact_exists')),
        'calibration_status_ok': str(calibration_diag.get('status', 'missing')) == 'ok',
        'calibration_samples_ok': int(calibration_diag.get('num_samples') or 0) >= int(CALIBRATION_MIN_SAMPLES),
    }
    failed_checks = [name for name, ok in checks.items() if not bool(ok)]
    return {
        'applicable': True,
        'pass': len(failed_checks) == 0,
        'failed_checks': failed_checks,
        'checks': checks,
        'minimums': {
            'walk_forward_samples': min_wf_samples,
            'calibration_samples': int(CALIBRATION_MIN_SAMPLES),
        },
    }


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


def _global_horizon_model_paths() -> Dict[int, str]:
    return {
        5: MODEL_PATH_5D,
        10: MODEL_PATH_10D,
        30: MODEL_PATH_30D,
    }


def _bucket_model_path(horizon: int, bucket: str) -> str:
    return os.path.join(BUCKET_MODEL_DIR, bucket.lower(), f"model_{horizon}d.pkl")


_BUCKET_SETS_CACHE: Optional[Dict[str, set]] = None
_INFERENCE_MODEL_CACHE: Dict[str, Any] = {}
_CALIBRATION_ARTIFACT_CACHE: Dict[int, Dict[str, object]] = {}


def _get_bucket_sets() -> Dict[str, set]:
    """Build static ticker sets used for asset-bucket model routing."""
    global _BUCKET_SETS_CACHE
    if _BUCKET_SETS_CACHE is not None:
        return _BUCKET_SETS_CACHE

    etf_set = set()
    bond_set = set()
    try:
        from universe import get_etf_universe, get_bond_universe

        etf_map = get_etf_universe()
        bond_map = get_bond_universe()
        for tickers in etf_map.values():
            etf_set.update(str(t).upper() for t in tickers if t)
        for tickers in bond_map.values():
            bond_set.update(str(t).upper() for t in tickers if t)
    except Exception as e:
        print(f"[bucket] Warning: Could not load universe maps for bucketing: {e}", flush=True)

    _BUCKET_SETS_CACHE = {
        'ETF': etf_set,
        'BOND': bond_set,
    }
    return _BUCKET_SETS_CACHE


def get_asset_bucket(ticker: str) -> str:
    """Return bucket label for a ticker."""
    t = (ticker or "").upper().strip()
    if not t:
        return "EQUITY"
    sets = _get_bucket_sets()
    if t in sets.get('BOND', set()):
        return "BOND"
    if t in sets.get('ETF', set()):
        return "ETF"
    return "EQUITY"


def _resolve_model_path_for_inference(horizon: int, ticker: Optional[str] = None) -> str:
    """Prefer bucket-specific model when available, else fall back to global horizon model."""
    global_paths = _global_horizon_model_paths()
    fallback = global_paths.get(horizon, MODEL_PATH)
    if not ticker:
        return fallback
    bucket = get_asset_bucket(ticker)
    if bucket != "EQUITY":
        candidate = _bucket_model_path(horizon, bucket)
        if os.path.exists(candidate):
            return candidate
    elif ASSET_BUCKET_MODELING_ENABLED and "EQUITY" in ASSET_BUCKETS:
        candidate = _bucket_model_path(horizon, "EQUITY")
        if os.path.exists(candidate):
            return candidate
    return fallback


def _load_model_cached(path: str):
    """Load model from disk once per path during inference."""
    cached = _INFERENCE_MODEL_CACHE.get(path)
    if cached is not None:
        return cached
    model = joblib.load(path)
    _INFERENCE_MODEL_CACHE[path] = model
    return model


def _compute_oos_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    top_quantile: float,
    top_k_frac: float,
    top_k_min_samples: int,
) -> Dict[str, float]:
    """Compute out-of-sample metrics focused on actionable high-confidence predictions."""
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(proba).astype(float)
    if y_arr.size == 0 or y_arr.size != p_arr.size:
        return {}

    pred_buy = p_arr >= DEFAULT_BUY_THRESHOLD
    accuracy = float(np.mean((pred_buy.astype(int) == y_arr.astype(int)))) if y_arr.size else 0.0
    base_hit_rate = float(np.mean(y_arr == 1)) if y_arr.size else 0.0

    auc = 0.5
    try:
        if np.unique(y_arr).size >= 2:
            auc = float(roc_auc_score(y_arr, p_arr))
    except Exception:
        auc = 0.5

    quantile = float(np.clip(top_quantile, 0.5, 0.99))
    cutoff = float(np.quantile(p_arr, quantile)) if p_arr.size else 1.0
    top_mask = p_arr >= cutoff
    selected = int(np.sum(top_mask))
    if selected == 0:
        top_hit_rate = 0.0
        top_avg_edge = 0.0
        top_sharpe_proxy = 0.0
    else:
        top_y = y_arr[top_mask]
        top_hit_rate = float(np.mean(top_y == 1))
        # +1 for wins, -1 for losses approximates directional edge.
        pnl = np.where(top_y == 1, 1.0, -1.0)
        top_avg_edge = float(np.mean(pnl))
        std = float(np.std(pnl, ddof=0))
        top_sharpe_proxy = float((np.mean(pnl) / std) * np.sqrt(252.0)) if std > 0 else 0.0

    k_frac = float(np.clip(top_k_frac, 0.01, 0.50))
    k_min = int(max(1, top_k_min_samples))
    k_count = int(max(k_min, round(len(p_arr) * k_frac)))
    k_count = int(min(len(p_arr), k_count))
    rank_idx = np.argsort(-p_arr)
    topk_idx = rank_idx[:k_count]
    topk_y = y_arr[topk_idx]

    precision_at_k = float(np.mean(topk_y == 1)) if topk_y.size else 0.0
    lift_at_k = float(precision_at_k / base_hit_rate) if base_hit_rate > 0 else 0.0
    if topk_y.size:
        topk_pnl = np.where(topk_y == 1, 1.0, -1.0)
        topk_avg_edge = float(np.mean(topk_pnl))
        topk_std = float(np.std(topk_pnl, ddof=0))
        topk_sharpe_proxy = float((np.mean(topk_pnl) / topk_std) * np.sqrt(252.0)) if topk_std > 0 else 0.0
    else:
        topk_avg_edge = 0.0
        topk_sharpe_proxy = 0.0

    return {
        'samples': float(y_arr.size),
        'buy_rate': float(np.mean(pred_buy)),
        'accuracy': accuracy,
        'auc': auc,
        'base_hit_rate': base_hit_rate,
        'top_quantile': quantile,
        'top_selected': float(selected),
        'top_hit_rate': top_hit_rate,
        'top_avg_edge': top_avg_edge,
        'top_sharpe_proxy': top_sharpe_proxy,
        'top_k_frac': k_frac,
        'top_k_count': float(k_count),
        'precision_at_k': precision_at_k,
        'lift_at_k': lift_at_k,
        'topk_avg_edge': topk_avg_edge,
        'topk_sharpe_proxy': topk_sharpe_proxy,
    }


def _build_date_grouped_splits(
    sample_dates: pd.Series,
    n_splits: int,
    gap_dates: int,
) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Build walk-forward splits grouped by trading date.

    All samples from the same date are kept in the same fold to prevent
    cross-sectional leakage between train/test in the same session.
    """
    if sample_dates is None or len(sample_dates) == 0:
        return None
    dt = pd.to_datetime(sample_dates, errors='coerce')
    if dt.isna().any():
        return None

    day = dt.dt.normalize()
    unique_days = np.array(sorted(day.unique()))
    if unique_days.size < (n_splits + 2):
        return None

    effective_gap = int(max(1, min(gap_dates, unique_days.size - 2)))
    try:
        splitter = TimeSeriesSplit(n_splits=n_splits, gap=effective_gap)
    except Exception:
        return None

    day_to_idx = {d: i for i, d in enumerate(unique_days)}
    sample_day_idx = day.map(day_to_idx).to_numpy()
    day_index = np.arange(unique_days.size)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for train_days, test_days in splitter.split(day_index):
        train_mask = np.isin(sample_day_idx, train_days)
        test_mask = np.isin(sample_day_idx, test_days)
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits or None


def _normalize_sample_intervals(sample_intervals: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Normalize t0/t1 interval metadata and enforce valid ordering."""
    if sample_intervals is None or not isinstance(sample_intervals, pd.DataFrame):
        return None
    if not {'t0', 't1'}.issubset(sample_intervals.columns):
        return None
    if len(sample_intervals) == 0:
        return None

    t0 = pd.to_datetime(sample_intervals['t0'], errors='coerce')
    t1 = pd.to_datetime(sample_intervals['t1'], errors='coerce').fillna(t0)
    valid = (~t0.isna()) & (~t1.isna())
    if not bool(valid.any()):
        return None
    t0 = t0.where(valid)
    t1 = t1.where(valid)
    t1 = t1.where(t1 >= t0, t0)
    return pd.DataFrame({'t0': t0, 't1': t1})


def _build_purged_walk_forward_splits(
    sample_intervals: Optional[pd.DataFrame],
    n_splits: int,
    gap_days: int,
    prefer_date_blocks: bool = True,
) -> Tuple[Optional[List[Tuple[np.ndarray, np.ndarray]]], Dict[str, float], str]:
    """
    Build walk-forward splits with interval purging and embargo handling.

    Purge rule:
      remove train samples where [t0, t1] overlaps any test interval.
    Embargo rule:
      remove train samples with t0 in (test_end, test_end + gap_days].
    """
    intervals = _normalize_sample_intervals(sample_intervals)
    stats: Dict[str, float] = {
        'train_candidates': 0.0,
        'train_kept': 0.0,
        'purged_overlap': 0.0,
        'purged_embargo': 0.0,
        'purged_total': 0.0,
        'purged_frac': 0.0,
        'max_overlap_after_purge': 0.0,
    }
    if intervals is None:
        return None, stats, "metadata_unavailable"

    split_mode = "purged_row_time"
    split_pairs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    if prefer_date_blocks:
        split_pairs = _build_date_grouped_splits(
            sample_dates=intervals['t0'],
            n_splits=n_splits,
            gap_dates=max(1, gap_days),
        )
        if split_pairs:
            split_mode = "purged_date_blocks"
    if split_pairs is None:
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits, gap=max(1, gap_days))
            split_pairs = list(tscv.split(np.arange(len(intervals))))
        except Exception:
            return None, stats, "metadata_unavailable"

    purged_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    embargo_delta = pd.Timedelta(days=max(1, gap_days))
    for train_idx, test_idx in split_pairs:
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        train_meta = intervals.iloc[train_idx]
        test_meta = intervals.iloc[test_idx]
        test_start = test_meta['t0'].min()
        test_end = test_meta['t1'].max()
        if pd.isna(test_start) or pd.isna(test_end):
            continue

        overlap_mask = (train_meta['t0'] <= test_end) & (train_meta['t1'] >= test_start)
        embargo_mask = (train_meta['t0'] > test_end) & (train_meta['t0'] <= (test_end + embargo_delta))
        extra_embargo = embargo_mask & (~overlap_mask)
        drop_mask = overlap_mask | embargo_mask

        kept_idx = train_idx[~drop_mask.to_numpy()]
        stats['train_candidates'] += float(len(train_idx))
        stats['train_kept'] += float(len(kept_idx))
        stats['purged_overlap'] += float(int(overlap_mask.sum()))
        stats['purged_embargo'] += float(int(extra_embargo.sum()))

        if len(kept_idx) == 0:
            continue

        remain_meta = intervals.iloc[kept_idx]
        overlap_after = int(((remain_meta['t0'] <= test_end) & (remain_meta['t1'] >= test_start)).sum())
        stats['max_overlap_after_purge'] = max(stats['max_overlap_after_purge'], float(overlap_after))
        purged_pairs.append((kept_idx, test_idx))

    if stats['train_candidates'] > 0:
        stats['purged_total'] = float(
            max(0.0, stats['train_candidates'] - stats['train_kept'])
        )
        stats['purged_frac'] = float(stats['purged_total'] / stats['train_candidates'])

    if not purged_pairs:
        return None, stats, split_mode
    return purged_pairs, stats, split_mode


def run_walk_forward_validation(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray],
    sample_dates: Optional[pd.Series],
    sample_intervals: Optional[pd.DataFrame],
    horizon: int,
    bucket: str,
) -> Dict[str, Any]:
    """Run strict OOS walk-forward validation with purged/embargo-aware splits."""
    min_total = WALK_FORWARD_MIN_TEST_SAMPLES * max(2, WALK_FORWARD_SPLITS)
    if len(X) < min_total:
        return {
            'bucket': bucket,
            'horizon': int(horizon),
            'skipped': True,
            'reason': f"insufficient_samples<{min_total}",
            'samples': int(len(X)),
        }

    n_splits = int(max(2, WALK_FORWARD_SPLITS))
    gap = int(max(1, horizon))
    split_mode = "row_time"
    purge_stats: Dict[str, float] = {
        'train_candidates': 0.0,
        'train_kept': 0.0,
        'purged_overlap': 0.0,
        'purged_embargo': 0.0,
        'purged_total': 0.0,
        'purged_frac': 0.0,
        'max_overlap_after_purge': 0.0,
    }
    split_pairs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

    if sample_intervals is not None:
        split_pairs, purge_stats, split_mode = _build_purged_walk_forward_splits(
            sample_intervals=sample_intervals,
            n_splits=n_splits,
            gap_days=gap,
            prefer_date_blocks=WALK_FORWARD_GROUP_BY_DATE,
        )

    if split_pairs is None and WALK_FORWARD_GROUP_BY_DATE and sample_dates is not None:
        split_pairs = _build_date_grouped_splits(sample_dates, n_splits=n_splits, gap_dates=gap)
        if split_pairs:
            split_mode = "date_blocks"
    if split_pairs is None:
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
            split_pairs = list(tscv.split(X))
        except Exception as e:
            return {
                'bucket': bucket,
                'horizon': int(horizon),
                'skipped': True,
                'reason': f"tscv_error:{e}",
                'samples': int(len(X)),
            }

    fold_metrics: List[Dict[str, float]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(split_pairs, 1):
        if len(test_idx) < WALK_FORWARD_MIN_TEST_SAMPLES:
            continue
        model = MarketModel()
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        try:
            if sample_weight is not None and len(sample_weight) == len(X):
                sw_train = np.asarray(sample_weight)[train_idx]
                model.model.fit(X_train, y_train, sample_weight=sw_train)
            else:
                model.model.fit(X_train, y_train)
            proba = model.model.predict_proba(X_test)[:, 1]
            metrics = _compute_oos_metrics(
                y_test.to_numpy(),
                proba,
                WALK_FORWARD_TOP_QUANTILE,
                WALK_FORWARD_TOP_K_FRAC,
                WALK_FORWARD_TOP_K_MIN_SAMPLES,
            )
            if metrics:
                metrics['fold'] = float(fold_idx)
                fold_metrics.append(metrics)
        except Exception as e:
            print(f"[wf] Fold {fold_idx} failed ({bucket}, {horizon}d): {e}", flush=True)

    if not fold_metrics:
        return {
            'bucket': bucket,
            'horizon': int(horizon),
            'skipped': True,
            'reason': "no_valid_folds",
            'samples': int(len(X)),
            'split_mode': split_mode,
            'purged_overlap': float(purge_stats.get('purged_overlap', 0.0)),
            'purged_embargo': float(purge_stats.get('purged_embargo', 0.0)),
            'purged_total': float(purge_stats.get('purged_total', 0.0)),
            'purged_frac': float(purge_stats.get('purged_frac', 0.0)),
            'max_overlap_after_purge': float(purge_stats.get('max_overlap_after_purge', 0.0)),
        }

    fold_df = pd.DataFrame(fold_metrics)
    weight = fold_df['samples'].clip(lower=1.0)
    weighted = lambda col: float(np.average(fold_df[col], weights=weight))
    return {
        'bucket': bucket,
        'horizon': int(horizon),
        'split_mode': split_mode,
        'skipped': False,
        'folds': int(len(fold_df)),
        'samples': int(len(X)),
        'oos_accuracy': weighted('accuracy'),
        'oos_auc': weighted('auc'),
        'oos_base_hit_rate': weighted('base_hit_rate'),
        'oos_buy_rate': weighted('buy_rate'),
        'oos_precision_at_k': weighted('precision_at_k'),
        'oos_lift_at_k': weighted('lift_at_k'),
        'oos_topk_avg_edge': weighted('topk_avg_edge'),
        'oos_topk_sharpe_proxy': weighted('topk_sharpe_proxy'),
        'oos_top_hit_rate': weighted('top_hit_rate'),
        'oos_top_avg_edge': weighted('top_avg_edge'),
        'oos_top_sharpe_proxy': weighted('top_sharpe_proxy'),
        'top_quantile': float(WALK_FORWARD_TOP_QUANTILE),
        'top_k_frac': float(WALK_FORWARD_TOP_K_FRAC),
        'top_k_min_samples': int(WALK_FORWARD_TOP_K_MIN_SAMPLES),
        'purged_overlap': float(purge_stats.get('purged_overlap', 0.0)),
        'purged_embargo': float(purge_stats.get('purged_embargo', 0.0)),
        'purged_total': float(purge_stats.get('purged_total', 0.0)),
        'purged_frac': float(purge_stats.get('purged_frac', 0.0)),
        'max_overlap_after_purge': float(purge_stats.get('max_overlap_after_purge', 0.0)),
    }


def _write_validation_report(entries: List[Dict[str, Any]]) -> None:
    """Persist walk-forward metrics for reproducible model comparisons."""
    os.makedirs(os.path.dirname(MODEL_VALIDATION_FILE), exist_ok=True)
    payload = {
        'generated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'promotion_gate_schema_version': 1,
        'entries': entries,
    }
    with open(MODEL_VALIDATION_FILE, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"[wf] Validation report updated: {MODEL_VALIDATION_FILE}", flush=True)


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
    qual_features: Dict[str, float] = extract_qual_features(qual_data)
    qual_features_frame = pd.DataFrame([qual_features], index=df.index)
    df = pd.concat([df, qual_features_frame], axis=1)

    
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
        df_clean = df.dropna().iloc[:-horizon].copy()

        # Label interval metadata for purged CV:
        # t0 = sample timestamp, t1 = end of forward label interval.
        t0_series = pd.to_datetime(
            df_clean.get('date', pd.Series(index=df_clean.index, dtype='datetime64[ns]')),
            errors='coerce'
        )
        raw_dates = pd.to_datetime(
            df.get('date', pd.Series(index=df.index, dtype='datetime64[ns]')),
            errors='coerce'
        )
        t1_values: List[pd.Timestamp] = []
        for idx in df_clean.index:
            t1_val = pd.NaT
            try:
                pos = int(idx) + int(horizon)
                if 0 <= pos < len(raw_dates):
                    t1_val = raw_dates.iloc[pos]
            except Exception:
                t1_val = pd.NaT
            if pd.isna(t1_val):
                base_dt = pd.to_datetime(df_clean.at[idx, 'date'], errors='coerce')
                if pd.notna(base_dt):
                    t1_val = base_dt + pd.Timedelta(days=int(horizon))
            t1_values.append(t1_val)
        df_clean['t0'] = t0_series
        df_clean['t1'] = pd.to_datetime(pd.Series(t1_values, index=df_clean.index), errors='coerce')
    else:
        # Inference mode: keep latest rows, no future-label trimming.
        df_clean = df.dropna()

    # === CONTRACT ENFORCEMENT & ALIGNMENT ===
    # Ensure correct columns and ORDER for ML inference/training:
    # add any missing feature columns in one pass (default 0.0) and preserve
    # contract order for downstream consumers.
    feature_block = df_clean.reindex(columns=ML_FEATURE_CONTRACT, fill_value=0.0)
    if for_inference:
        return feature_block
    return pd.concat([feature_block, df_clean[['meta_label', 't0', 't1']]], axis=1)


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
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'date', 'ticker', 'meta_label', 'signal', 't0', 't1']
        feature_cols = [c for c in features.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(features[c])]
        
        if 'meta_label' in features.columns and len(feature_cols) > 10:
            result = features[feature_cols + ['meta_label']].copy()
            if 't0' in features.columns:
                result['t0'] = pd.to_datetime(features['t0'], errors='coerce').values
                result['date'] = pd.to_datetime(features['t0'], errors='coerce').values
            if 't1' in features.columns:
                result['t1'] = pd.to_datetime(features['t1'], errors='coerce').values
            if 'date' in df.columns and 'date' not in result.columns:
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
                          work_units_done: float = None, return_weights: bool = False,
                          return_dates: bool = False,
                          return_intervals: bool = False):
    """
    Loads data, generates signals, and prepares X/y for training.
    Uses parallel processing to maximize CPU utilization on M1 Ultra.
    
    Args:
        tickers: List of tickers to process
        tracker: ETA tracker for status updates
        status_file: Path to status file
        horizon: Prediction horizon in days (5, 10, or 30)
        return_intervals: when True, also return interval metadata (t0/t1) for purged CV.
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

    def _empty_training_output():
        empty_x = pd.DataFrame()
        empty_y = pd.Series(dtype='float64')
        empty_w = np.array([])
        empty_dates = pd.Series(dtype='datetime64[ns]')
        empty_intervals = pd.DataFrame({'t0': pd.Series(dtype='datetime64[ns]'),
                                        't1': pd.Series(dtype='datetime64[ns]')})
        if return_weights and return_dates and return_intervals:
            return empty_x, empty_y, empty_w, empty_dates, empty_intervals
        if return_weights and return_dates:
            return empty_x, empty_y, empty_w, empty_dates
        if return_weights and return_intervals:
            return empty_x, empty_y, empty_w, empty_intervals
        if return_dates and return_intervals:
            return empty_x, empty_y, empty_dates, empty_intervals
        if return_weights:
            return empty_x, empty_y, empty_w
        if return_dates:
            return empty_x, empty_y, empty_dates
        if return_intervals:
            return empty_x, empty_y, empty_intervals
        return empty_x, empty_y

    if not all_features:
        return _empty_training_output()
    
    print(f"  Concatenating {len(all_features)} feature sets...", flush=True)
    try:
        data = pd.concat(all_features)
    except ValueError:
        # Handle case where all results were empty/None logic fell through
        return _empty_training_output()
    
    # Ensure chronological ordering and build interval metadata for purged CV.
    sample_weights = None
    sample_dates = pd.Series(dtype='datetime64[ns]')
    sample_intervals = pd.DataFrame({'t0': pd.Series(dtype='datetime64[ns]'),
                                     't1': pd.Series(dtype='datetime64[ns]')})

    sort_col = None
    if 'date' in data.columns:
        sort_col = 'date'
    elif 't0' in data.columns:
        sort_col = 't0'
    if sort_col is not None:
        data = data.sort_values(sort_col).reset_index(drop=True)
        sample_dates = pd.to_datetime(data[sort_col], errors='coerce')
    if return_weights:
        if RECENCY_WEIGHTING_ENABLED and sort_col is not None:
            sample_weights = _compute_recency_weights(data[sort_col])
        else:
            sample_weights = np.ones(len(data))

    if 't0' in data.columns and 't1' in data.columns:
        t0 = pd.to_datetime(data['t0'], errors='coerce')
        t1 = pd.to_datetime(data['t1'], errors='coerce').fillna(t0)
        t1 = t1.where(t1 >= t0, t0)
        sample_intervals = pd.DataFrame({'t0': t0, 't1': t1})

    drop_meta_cols = [c for c in ('date', 't0', 't1') if c in data.columns]
    if drop_meta_cols:
        data = data.drop(columns=drop_meta_cols)

    # CRITICAL: Numerical Stability Cleanup
    # Replace infinity with NaN, then fill all NaNs with 0
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Clip extreme values to prevent 'value too large for float32' errors
    # Most technical indicators and macro ratios shouldn't be in the millions.
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].clip(lower=-1e6, upper=1e6)

    # Ensure all feature columns exist and are contract-ordered, defaulting missing
    # values to 0.0 in a single batched step.
    X_out = data.reindex(columns=FEATURE_COLS, fill_value=0.0)

    y_out = data['meta_label']
    if return_weights and return_dates and return_intervals:
        if sample_weights is None or len(sample_weights) != len(X_out):
            sample_weights = np.ones(len(X_out))
        if sample_dates.empty or len(sample_dates) != len(X_out):
            sample_dates = pd.Series(pd.NaT, index=X_out.index)
        if sample_intervals.empty or len(sample_intervals) != len(X_out):
            sample_intervals = pd.DataFrame({'t0': sample_dates, 't1': sample_dates})
        return X_out, y_out, sample_weights, sample_dates, sample_intervals
    if return_weights and return_dates:
        if sample_weights is None or len(sample_weights) != len(X_out):
            sample_weights = np.ones(len(X_out))
        if sample_dates.empty or len(sample_dates) != len(X_out):
            sample_dates = pd.Series(pd.NaT, index=X_out.index)
        return X_out, y_out, sample_weights, sample_dates
    if return_weights and return_intervals:
        if sample_weights is None or len(sample_weights) != len(X_out):
            sample_weights = np.ones(len(X_out))
        if sample_intervals.empty or len(sample_intervals) != len(X_out):
            sample_intervals = pd.DataFrame({'t0': pd.Series(pd.NaT, index=X_out.index),
                                             't1': pd.Series(pd.NaT, index=X_out.index)})
        return X_out, y_out, sample_weights, sample_intervals
    if return_dates and return_intervals:
        if sample_dates.empty or len(sample_dates) != len(X_out):
            sample_dates = pd.Series(pd.NaT, index=X_out.index)
        if sample_intervals.empty or len(sample_intervals) != len(X_out):
            sample_intervals = pd.DataFrame({'t0': sample_dates, 't1': sample_dates})
        return X_out, y_out, sample_dates, sample_intervals
    if return_weights:
        if sample_weights is None or len(sample_weights) != len(X_out):
            sample_weights = np.ones(len(X_out))
        return X_out, y_out, sample_weights
    if return_dates:
        if sample_dates.empty or len(sample_dates) != len(X_out):
            sample_dates = pd.Series(pd.NaT, index=X_out.index)
        return X_out, y_out, sample_dates
    if return_intervals:
        if sample_intervals.empty or len(sample_intervals) != len(X_out):
            sample_intervals = pd.DataFrame({'t0': pd.Series(pd.NaT, index=X_out.index),
                                             't1': pd.Series(pd.NaT, index=X_out.index)})
        return X_out, y_out, sample_intervals
    return X_out, y_out


def _build_training_groups(tickers: List[str]) -> List[Tuple[str, List[str]]]:
    """Build model-training groups: global plus optional bucket-specific cohorts."""
    groups: List[Tuple[str, List[str]]] = [("GLOBAL", tickers)]
    if not ASSET_BUCKET_MODELING_ENABLED:
        return groups

    bucket_map: Dict[str, List[str]] = {b: [] for b in ASSET_BUCKETS}
    for ticker in tickers:
        bucket = get_asset_bucket(ticker)
        if bucket in bucket_map:
            bucket_map[bucket].append(ticker)

    for bucket in ASSET_BUCKETS:
        bucket_tickers = bucket_map.get(bucket, [])
        if len(bucket_tickers) >= ASSET_BUCKET_MIN_TICKERS:
            groups.append((bucket, bucket_tickers))
        else:
            print(
                f"[bucket] Skipping {bucket}: {len(bucket_tickers)} tickers "
                f"(min={ASSET_BUCKET_MIN_TICKERS})",
                flush=True,
            )
    return groups


def train_market_model():
    """
    Train global + bucket models across all horizons with strict walk-forward OOS validation.
    """
    try:
        from storage import get_existing_tickers
    except ImportError:
        src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from storage import get_existing_tickers

    from eta_tracker import ETATracker
    from config import PROJECT_ROOT

    status_file = os.path.join(PROJECT_ROOT, 'TRAIN_STATUS.md')
    tracker = ETATracker("ml_training")
    tickers = get_existing_tickers()
    training_groups = _build_training_groups(tickers)
    total_jobs = max(1, len(HORIZONS) * len(training_groups))

    print(f"Training on {len(tickers)} tickers across {len(HORIZONS)} horizons.", flush=True)
    print(f"Training groups: {[name for name, _ in training_groups]}", flush=True)
    print(f"Models: RF + GBM" + (" + XGBoost" if HAS_XGBOOST else ""), flush=True)

    tracker.start_run({"tickers": len(tickers), "horizons": len(HORIZONS), "groups": len(training_groups)})
    total_est_samples = sum(max(1, len(group_tickers)) * 1000 for _, group_tickers in training_groups) * len(HORIZONS)
    tracker.set_work_units(total_est_samples)

    session_tracker = ETATracker("ml_session")
    session_tracker.start_run({"total_jobs": total_jobs})
    session_tracker.set_work_units(total_est_samples)
    tracker.session_tracker = session_tracker

    tracker.write_status(0.01, "Initializing Multi-Horizon Training", status_file)

    validation_entries: List[Dict[str, Any]] = []
    saved_paths: List[str] = []
    global_paths = _global_horizon_model_paths()

    for horizon_idx, horizon in enumerate(HORIZONS):
        for group_idx, (bucket, group_tickers) in enumerate(training_groups):
            job_idx = horizon_idx * len(training_groups) + group_idx
            progress_base = job_idx / total_jobs
            progress_step = 1.0 / total_jobs
            horizon_info = {
                'current_horizon': horizon,
                'horizon_idx': horizon_idx,
                'total_horizons': len(HORIZONS),
            }

            print(f"\n{'='*60}", flush=True)
            print(
                f"Training {horizon}d | {bucket} "
                f"({job_idx + 1}/{total_jobs}, tickers={len(group_tickers)})",
                flush=True,
            )
            print(f"{'='*60}", flush=True)

            tracker.write_status(
                progress_base + 0.01,
                f"Training {horizon}d {bucket}",
                status_file,
                horizon_info=horizon_info,
            )

            cv_samples_per_job = max(1, len(group_tickers)) * 1000
            cum_work_done = job_idx * cv_samples_per_job
            X, y, sample_weights, sample_dates, sample_intervals = prepare_training_data(
                group_tickers,
                tracker=tracker,
                status_file=status_file,
                horizon=horizon,
                horizon_info=horizon_info,
                work_units_done=cum_work_done,
                return_weights=True,
                return_dates=True,
                return_intervals=True,
            )

            if X.empty:
                print(f"  WARNING: No data for {horizon}d {bucket}. Skipping.", flush=True)
                continue
            if bucket != "GLOBAL" and len(X) < ASSET_BUCKET_MIN_SAMPLES:
                print(
                    f"  WARNING: {bucket} {horizon}d only has {len(X)} samples "
                    f"(min={ASSET_BUCKET_MIN_SAMPLES}). Skipping.",
                    flush=True,
                )
                continue

            print(f"  Data: {X.shape}, Classes: {y.value_counts().to_dict()}", flush=True)
            model = MarketModel()
            model_save_path = global_paths[horizon] if bucket == "GLOBAL" else _bucket_model_path(horizon, bucket)
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            model.train(
                X,
                y,
                tracker=tracker,
                status_file=status_file,
                horizon_info=horizon_info,
                save_path=model_save_path,
                sample_weight=sample_weights,
                save_thresholds_for_horizon=(bucket == "GLOBAL"),
                sample_intervals=sample_intervals,
            )
            saved_paths.append(model_save_path)

            if bucket == "GLOBAL":
                threshold_diag = _get_threshold_diagnostics(horizon)
                calibration_diag = _get_calibration_diagnostics(horizon, model_path=model_save_path)
            else:
                threshold_diag = {
                    'horizon': int(horizon),
                    'source': 'not_applicable',
                }
                calibration_diag = {
                    'horizon': int(horizon),
                    'artifact_path': '',
                    'artifact_exists': False,
                    'status': 'not_applicable',
                    'method': None,
                    'num_samples': 0,
                    'num_valid_samples': 0,
                    'bin_count': 0,
                    'generated_at': None,
                    'model_sha256': None,
                    'model_sha_match': None,
                }

            if WALK_FORWARD_ENABLED:
                wf_result = run_walk_forward_validation(
                    X=X,
                    y=y,
                    sample_weight=sample_weights,
                    sample_dates=sample_dates,
                    sample_intervals=sample_intervals,
                    horizon=horizon,
                    bucket=bucket,
                )
                wf_result['threshold_diagnostics'] = threshold_diag
                wf_result['calibration_diagnostics'] = calibration_diag
                wf_result['promotion_gates'] = _build_promotion_gate_status(
                    wf_result=wf_result,
                    threshold_diag=threshold_diag,
                    calibration_diag=calibration_diag,
                )
                validation_entries.append(wf_result)
                print(f"[wf] {bucket} {horizon}d -> {wf_result}", flush=True)

            tracker.write_status(
                progress_base + progress_step * 0.95,
                f"Completed {horizon}d {bucket}",
                status_file,
                horizon_info=horizon_info,
            )

    if validation_entries:
        _write_validation_report(validation_entries)

    tracker.write_status(1.0, "All Horizons Complete", status_file)
    tracker.end_run()

    print(f"\n{'='*60}", flush=True)
    print("MULTI-HORIZON TRAINING COMPLETE", flush=True)
    print(f"Models saved: {saved_paths}", flush=True)
    if validation_entries:
        print(f"Validation metrics: {MODEL_VALIDATION_FILE}", flush=True)
    print(f"{'='*60}", flush=True)


def get_signal_confidence(ticker_row: pd.Series, ticker: Optional[str] = None) -> float:
    """Get ML confidence for a single ticker's latest data (10d model for backward compatibility)."""
    available_cols = [c for c in FEATURE_COLS if c in ticker_row.index]
    if len(available_cols) < 5:
        return 0.5
    confs = get_multi_horizon_confidence(ticker_row, ticker=ticker)
    return float(confs.get('conf_10d', 0.5))


def get_multi_horizon_confidence(ticker_row: pd.Series, ticker: Optional[str] = None) -> dict:
    """
    Get ML confidence for all 3 prediction horizons.
    
    Returns:
        dict with keys: 'conf_5d', 'conf_10d', 'conf_30d' plus raw values
        (for example 'raw_conf_10d', 'raw_conf_5d', 'raw_conf_30d') and
        calibration flags (for example 'calibration_conf_10d').
    """
    available_cols = [c for c in FEATURE_COLS if c in ticker_row.index]
    if len(available_cols) < 5:
        return {
            'conf_5d': 0.5,
            'conf_10d': 0.5,
            'conf_30d': 0.5,
            'raw_conf_5d': 0.5,
            'raw_conf_10d': 0.5,
            'raw_conf_30d': 0.5,
            'calibration_conf_5d': False,
            'calibration_conf_10d': False,
            'calibration_conf_30d': False,
        }
    
    # Fill missing columns with 0
    X_dict = {c: ticker_row.get(c, 0) for c in FEATURE_COLS}
    X = pd.DataFrame([X_dict])[FEATURE_COLS]
    
    results = {}
    
    # Model paths for each horizon
    horizon_paths = {
        'conf_5d': _resolve_model_path_for_inference(5, ticker=ticker),
        'conf_10d': _resolve_model_path_for_inference(10, ticker=ticker),
        'conf_30d': _resolve_model_path_for_inference(30, ticker=ticker),
    }

    horizon_values = {5: 5, 10: 10, 30: 30}

    for key, path in horizon_paths.items():
        horizon = horizon_values.get(int(key.replace('conf_', '').replace('d', '')))
        try:
            if os.path.exists(path):
                model = _load_model_cached(path)
                proba = model.predict_proba(X)[:, 1]
                raw_proba = float(np.clip(proba[0], 0, 1))
                if horizon is None:
                    results[key] = raw_proba
                    continue
                results[f'raw_{key}'] = raw_proba
                results[key] = _apply_probability_calibration(raw_proba, horizon)
                results[f'calibration_{key}'] = bool(_load_probability_calibration(horizon) is not None)
            else:
                # Model not yet trained
                results[key] = 0.5
                results[f'raw_{key}'] = 0.5
                results[f'calibration_{key}'] = False
        except Exception as e:
            print(f"  Warning: Could not load {key} model: {e}", flush=True)
            results[key] = 0.5
            results[f'raw_{key}'] = 0.5
            results[f'calibration_{key}'] = False
    
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
