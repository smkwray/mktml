import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import json

# Calculate PROJECT_ROOT based on script location (audit.py is in src/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_SCRIPT_DIR) == 'src':
    PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
else:
    PROJECT_ROOT = _SCRIPT_DIR

# Add project root and src to path for imports
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from storage import get_connection, load_price_data, save_price_data
try:
    from config import (
        AUDIT_BENCHMARK_TICKER,
        AUDIT_FORWARD_DAYS,
        AUDIT_STALE_CLOSE_DAYS,
        AUDIT_BLOCK_ON_STALE_OPEN_ROWS,
        AUDIT_MAX_STALE_OPEN_ROWS,
        AUDIT_MIN_DIRECTIONAL_SAMPLES,
        AUDIT_MIN_BUY_SAMPLES,
        AUDIT_MIN_BENCHMARK_DATES,
        AUDIT_MIN_CONF_BUCKET_SAMPLES,
        BENCHMARK_BLEND,
        DEFAULT_BUY_THRESHOLD,
        DEFAULT_SELL_THRESHOLD,
        PUBLIC_REPORTS_DIR,
        get_model_thresholds,
    )
except ImportError:
    AUDIT_BENCHMARK_TICKER = 'SPY'
    AUDIT_FORWARD_DAYS = 5
    AUDIT_STALE_CLOSE_DAYS = 45
    AUDIT_BLOCK_ON_STALE_OPEN_ROWS = True
    AUDIT_MAX_STALE_OPEN_ROWS = 0
    AUDIT_MIN_DIRECTIONAL_SAMPLES = 200
    AUDIT_MIN_BUY_SAMPLES = 30
    AUDIT_MIN_BENCHMARK_DATES = 20
    AUDIT_MIN_CONF_BUCKET_SAMPLES = 10
    BENCHMARK_BLEND = {'SPY': 0.6, 'AGG': 0.4}
    DEFAULT_BUY_THRESHOLD = 0.60
    DEFAULT_SELL_THRESHOLD = 0.40
    PUBLIC_REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'public')
    def get_model_thresholds(horizon: Optional[int] = None) -> Dict[str, float]:
        return {'buy': DEFAULT_BUY_THRESHOLD, 'sell': DEFAULT_SELL_THRESHOLD}

def calculate_metrics(series: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    """Calculate CAGR, Sharpe, MaxDD, and annualized volatility."""
    if series.empty:
        return {'cagr': 0, 'sharpe': 0, 'max_dd': 0, 'vol': 0}
    
    # CAGR
    total_ret = (1 + series).prod() - 1
    days = len(series)
    if days < 1: return {'cagr': 0, 'sharpe': 0, 'max_dd': 0, 'vol': 0}
    cagr = (1 + total_ret) ** (periods_per_year / days) - 1
    
    # Sharpe (Assume Rf=0 for simplicity)
    mean_ret = series.mean() * periods_per_year
    vol = series.std() * np.sqrt(periods_per_year)
    sharpe = mean_ret / vol if vol != 0 else 0
    
    # Max Drawdown
    cum_ret = (1 + series).cumprod()
    peak = cum_ret.cummax()
    dd = (cum_ret - peak) / peak
    max_dd = dd.min()
    
    return {
        'cagr': cagr,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'vol': vol
    }

def _sanitize_token(text: str) -> str:
    token = ''.join(ch.lower() if ch.isalnum() else '_' for ch in text.strip())
    return token.strip('_') or 'benchmark'


def _compute_forward_returns(ticker: str, horizon_days: int, force_refresh: bool = False) -> pd.DataFrame:
    """Compute horizon forward returns per date for a benchmark ticker."""
    prices = load_price_data(ticker)
    if prices.empty or force_refresh:
        try:
            # Try full fallback chain (yfinance -> multiple APIs) before giving up.
            from data_loader import download_batch_with_fallback
            fresh = download_batch_with_fallback([ticker])
            if not fresh.empty:
                save_price_data(fresh)
                prices = load_price_data(ticker)
        except Exception as e:
            print(f"  [audit] Could not refresh benchmark ticker {ticker}: {e}")
    if prices.empty:
        return pd.DataFrame(columns=['date', 'benchmark_return'])

    df = prices[['date', 'close']].copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.dropna(subset=['close']).sort_values('date')
    df['benchmark_return'] = (df['close'].shift(-horizon_days) / df['close']) - 1.0
    return df[['date', 'benchmark_return']].dropna()


def _build_prediction_vs_reference_curve(
    perf_recs: pd.DataFrame,
    reference_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Build strategy-vs-reference curve from precomputed reference returns."""
    if perf_recs.empty or reference_returns.empty:
        return pd.DataFrame()

    buy_recs = perf_recs[(perf_recs['signal_type'] == 'BUY') & (perf_recs['perf_1w'].notna())].copy()
    if buy_recs.empty:
        return pd.DataFrame()

    buy_recs['date'] = pd.to_datetime(buy_recs['date']).dt.normalize()
    strategy_returns = buy_recs.groupby('date')['perf_1w'].mean().rename('strategy_return').reset_index()

    curve = strategy_returns.merge(reference_returns, on='date', how='inner').sort_values('date')
    if curve.empty:
        return pd.DataFrame()

    curve['excess_return'] = curve['strategy_return'] - curve['benchmark_return']
    curve['strategy_curve'] = (1.0 + curve['strategy_return']).cumprod()
    curve['benchmark_curve'] = (1.0 + curve['benchmark_return']).cumprod()
    curve['excess_curve'] = curve['strategy_curve'] / curve['benchmark_curve']
    return curve


def _compute_blended_forward_returns(
    horizon_days: int,
    blend_weights: Dict[str, float],
) -> pd.DataFrame:
    """Compute weighted forward returns for configured benchmark blend."""
    if not blend_weights:
        return pd.DataFrame(columns=['date', 'benchmark_return'])

    frames = []
    valid_weights = {}
    for ticker, weight in blend_weights.items():
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        if w <= 0:
            continue
        ticker_returns = _compute_forward_returns(ticker, horizon_days)
        if ticker_returns.empty:
            print(f"  [audit] Blend component {ticker} missing forward returns; skipping.")
            continue
        col = f"ret_{_sanitize_token(ticker)}"
        frames.append(ticker_returns.rename(columns={'benchmark_return': col}))
        valid_weights[col] = w

    if not frames:
        return pd.DataFrame(columns=['date', 'benchmark_return'])

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on='date', how='inner')
        if merged.empty:
            break

    if merged.empty:
        return pd.DataFrame(columns=['date', 'benchmark_return'])

    total_weight = sum(valid_weights.values())
    if total_weight <= 0:
        return pd.DataFrame(columns=['date', 'benchmark_return'])

    weighted_sum = np.zeros(len(merged), dtype=float)
    for col, weight in valid_weights.items():
        weighted_sum += merged[col].to_numpy() * weight
    merged['benchmark_return'] = weighted_sum / total_weight
    return merged[['date', 'benchmark_return']]


def _build_prediction_vs_benchmark_curve(
    perf_recs: pd.DataFrame,
    benchmark_ticker: str,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Build date-aligned performance curve:
    - Strategy: equal-weight mean of BUY prediction 1-week forward returns by date
    - Benchmark: horizon forward return of benchmark_ticker on the same dates
    """
    benchmark_returns = _compute_forward_returns(benchmark_ticker, horizon_days)
    if perf_recs.empty:
        return pd.DataFrame()
    buy_recs = perf_recs[(perf_recs['signal_type'] == 'BUY') & (perf_recs['perf_1w'].notna())].copy()
    if buy_recs.empty:
        return pd.DataFrame()
    buy_recs['date'] = pd.to_datetime(buy_recs['date']).dt.normalize()
    strategy_returns = buy_recs.groupby('date')['perf_1w'].mean().rename('strategy_return').reset_index()
    if not benchmark_returns.empty:
        strategy_end = strategy_returns['date'].max()
        benchmark_end = benchmark_returns['date'].max()
        required_end = strategy_end + timedelta(days=horizon_days + 2)
        if benchmark_end < required_end:
            # Benchmark looks stale for current signal range; refresh once and retry.
            refreshed = _compute_forward_returns(benchmark_ticker, horizon_days, force_refresh=True)
            if not refreshed.empty:
                benchmark_returns = refreshed
    if benchmark_returns.empty:
        print(f"  [audit] Benchmark ticker {benchmark_ticker} not found in DB for comparison.")
        return pd.DataFrame()

    return _build_prediction_vs_reference_curve(perf_recs=perf_recs, reference_returns=benchmark_returns)


def _build_prediction_vs_blend_curve(
    perf_recs: pd.DataFrame,
    horizon_days: int,
    blend_weights: Dict[str, float],
) -> pd.DataFrame:
    """Build date-aligned performance curve against blended benchmark."""
    if perf_recs.empty:
        return pd.DataFrame()
    blend_returns = _compute_blended_forward_returns(horizon_days, blend_weights)
    if blend_returns.empty:
        print("  [audit] Blended benchmark unavailable (missing component history overlap).")
        return pd.DataFrame()
    return _build_prediction_vs_reference_curve(perf_recs=perf_recs, reference_returns=blend_returns)


def _export_prediction_vs_benchmark(
    curve: pd.DataFrame,
    benchmark_ticker: str,
    horizon_days: int,
) -> Optional[Dict[str, str]]:
    """Export benchmark-relative curve to CSV and markdown summary."""
    if curve.empty:
        return None

    reports_dir = os.path.join(PROJECT_ROOT, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    benchmark_token = _sanitize_token(benchmark_ticker)

    csv_path = os.path.join(reports_dir, f'prediction_vs_{benchmark_token}.csv')
    md_path = os.path.join(reports_dir, f'prediction_vs_{benchmark_token}.md')

    to_save = curve.copy()
    to_save['date'] = pd.to_datetime(to_save['date']).dt.strftime('%Y-%m-%d')
    to_save.to_csv(csv_path, index=False)

    strategy_metrics = calculate_metrics(curve['strategy_return'], periods_per_year=52)
    benchmark_metrics = calculate_metrics(curve['benchmark_return'], periods_per_year=52)
    excess_std = curve['excess_return'].std()
    information_ratio = (
        (curve['excess_return'].mean() / excess_std) * np.sqrt(52)
        if excess_std and excess_std > 0
        else 0.0
    )
    beat_rate = float((curve['excess_return'] > 0).mean())

    with open(md_path, 'w') as f:
        f.write(f"# Prediction Performance vs {benchmark_ticker}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Uses BUY recommendations and {horizon_days}-trading-day forward returns.\n\n")
        f.write("| Metric | Strategy | Benchmark |\n")
        f.write("|--------|----------|-----------|\n")
        f.write(f"| Total Return | {(curve['strategy_curve'].iloc[-1] - 1):.2%} | {(curve['benchmark_curve'].iloc[-1] - 1):.2%} |\n")
        f.write(f"| CAGR (approx) | {strategy_metrics['cagr']:.2%} | {benchmark_metrics['cagr']:.2%} |\n")
        f.write(f"| Sharpe (approx) | {strategy_metrics['sharpe']:.2f} | {benchmark_metrics['sharpe']:.2f} |\n")
        f.write(f"| Max Drawdown | {strategy_metrics['max_dd']:.2%} | {benchmark_metrics['max_dd']:.2%} |\n")
        f.write(f"| Beat Rate | {beat_rate:.1%} | n/a |\n")
        f.write(f"| Information Ratio | {information_ratio:.2f} | n/a |\n\n")
        f.write("## Recent Periods\n\n")
        f.write("| Date | Strategy | Benchmark | Excess | Strat Curve | Bench Curve |\n")
        f.write("|------|----------|-----------|--------|-------------|-------------|\n")
        for _, row in curve.tail(20).iterrows():
            f.write(
                f"| {pd.to_datetime(row['date']).strftime('%Y-%m-%d')} | "
                f"{row['strategy_return']:.2%} | {row['benchmark_return']:.2%} | "
                f"{row['excess_return']:.2%} | {row['strategy_curve']:.3f} | {row['benchmark_curve']:.3f} |\n"
            )

    return {'csv_path': csv_path, 'md_path': md_path}


def _build_confidence_bucket_stats(perf_recs: pd.DataFrame) -> pd.DataFrame:
    """Build confidence-bucket outcomes for BUY signals."""
    if perf_recs.empty:
        return pd.DataFrame(columns=['bucket', 'count', 'avg_return', 'win_rate'])

    df = perf_recs[(perf_recs['signal_type'] == 'BUY') & (perf_recs['perf_1w'].notna())].copy()
    if df.empty:
        return pd.DataFrame(columns=['bucket', 'count', 'avg_return', 'win_rate'])

    bins = [0.0, 0.55, 0.65, 0.75, 0.85, 1.0]
    labels = ['<=55%', '55-65%', '65-75%', '75-85%', '85-100%']
    df['bucket'] = pd.cut(df['confidence'].clip(0, 1), bins=bins, labels=labels, include_lowest=True)
    grouped = df.groupby('bucket', observed=True).agg(
        count=('perf_1w', 'size'),
        avg_return=('perf_1w', 'mean'),
        win_rate=('perf_1w', lambda s: float((s > 0).mean())),
    ).reset_index()
    grouped = grouped[grouped['count'] >= AUDIT_MIN_CONF_BUCKET_SAMPLES].copy()
    grouped['bucket'] = grouped['bucket'].astype(str)
    return grouped


def _build_sample_adequacy_flags(
    directional_metrics: Dict[str, Any],
    buy_only_metrics: Dict[str, Any],
    benchmark_curve: pd.DataFrame,
) -> Dict[str, Any]:
    """Return minimum-sample reliability flags for audit interpretations."""
    directional_n = int((directional_metrics or {}).get('sample_count', 0) or 0)
    buy_n = int((buy_only_metrics or {}).get('buy_count', 0) or 0)
    benchmark_n = int(len(benchmark_curve)) if benchmark_curve is not None else 0
    return {
        'directional_sample_count': directional_n,
        'buy_sample_count': buy_n,
        'benchmark_aligned_dates': benchmark_n,
        'directional_sample_ok': directional_n >= AUDIT_MIN_DIRECTIONAL_SAMPLES,
        'buy_sample_ok': buy_n >= AUDIT_MIN_BUY_SAMPLES,
        'benchmark_sample_ok': benchmark_n >= AUDIT_MIN_BENCHMARK_DATES,
        'thresholds': {
            'directional_min': int(AUDIT_MIN_DIRECTIONAL_SAMPLES),
            'buy_min': int(AUDIT_MIN_BUY_SAMPLES),
            'benchmark_dates_min': int(AUDIT_MIN_BENCHMARK_DATES),
            'confidence_bucket_min': int(AUDIT_MIN_CONF_BUCKET_SAMPLES),
        },
    }


def _build_directional_signal_metrics(
    perf_recs: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
) -> Dict[str, Any]:
    """Compute realized directional quality for BUY/SELL/NEUTRAL calls."""
    if perf_recs.empty:
        return {}

    df = perf_recs[perf_recs['perf_1w'].notna() & perf_recs['confidence'].notna()].copy()
    if df.empty:
        return {}

    df['pred_dir'] = np.where(
        df['confidence'] >= buy_threshold,
        'BUY',
        np.where(df['confidence'] <= sell_threshold, 'SELL', 'NEUTRAL'),
    )
    df['actual_up'] = df['perf_1w'] > 0

    buy_df = df[df['pred_dir'] == 'BUY']
    sell_df = df[df['pred_dir'] == 'SELL']
    directional_df = df[df['pred_dir'] != 'NEUTRAL']

    sell_win_rate = float((sell_df['perf_1w'] < 0).mean()) if not sell_df.empty else 0.0
    directional_accuracy = float(
        ((directional_df['pred_dir'] == 'BUY') == directional_df['actual_up']).mean()
    ) if not directional_df.empty else 0.0

    return {
        'sample_count': int(len(df)),
        'buy_calls': int(len(buy_df)),
        'sell_calls': int(len(sell_df)),
        'neutral_calls': int((df['pred_dir'] == 'NEUTRAL').sum()),
        'buy_win_rate': float((buy_df['perf_1w'] > 0).mean()) if not buy_df.empty else 0.0,
        'sell_win_rate': sell_win_rate,
        'directional_accuracy': directional_accuracy,
        'buy_avg_return': float(buy_df['perf_1w'].mean()) if not buy_df.empty else 0.0,
        'sell_avg_return': float(sell_df['perf_1w'].mean()) if not sell_df.empty else 0.0,
    }


def _build_monthly_relative(curve: pd.DataFrame) -> pd.DataFrame:
    """Aggregate strategy/benchmark/excess returns by month."""
    if curve.empty:
        return pd.DataFrame(columns=['month', 'strategy_return', 'benchmark_return', 'excess_return'])
    df = curve.copy()
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M').astype(str)
    monthly = (
        df.groupby('month')[['strategy_return', 'benchmark_return', 'excess_return']]
        .apply(lambda x: (1.0 + x).prod() - 1.0)
        .reset_index()
    )
    return monthly.sort_values('month')


def _build_buy_only_metrics(
    perf_recs: pd.DataFrame,
    benchmark_curve: pd.DataFrame,
    blend_curve: Optional[pd.DataFrame],
    buy_threshold: float,
) -> Dict[str, Any]:
    """Compute BUY-only realized performance metrics."""
    if perf_recs.empty:
        return {}

    buys = perf_recs[(perf_recs['signal_type'] == 'BUY') & (perf_recs['perf_1w'].notna())].copy()
    if buys.empty:
        return {}

    high_conf = buys[buys['confidence'] >= buy_threshold].copy()
    metrics = {
        'buy_count': int(len(buys)),
        'buy_win_rate': float((buys['perf_1w'] > 0).mean()),
        'buy_avg_return': float(buys['perf_1w'].mean()),
        'buy_median_return': float(buys['perf_1w'].median()),
        'buy_profit_factor': float(
            buys[buys['perf_1w'] > 0]['perf_1w'].sum() / abs(buys[buys['perf_1w'] < 0]['perf_1w'].sum())
        ) if abs(buys[buys['perf_1w'] < 0]['perf_1w'].sum()) > 0 else 0.0,
        'high_conf_buy_count': int(len(high_conf)),
        'high_conf_buy_win_rate': float((high_conf['perf_1w'] > 0).mean()) if not high_conf.empty else 0.0,
        'high_conf_buy_avg_return': float(high_conf['perf_1w'].mean()) if not high_conf.empty else 0.0,
    }

    if not benchmark_curve.empty:
        metrics['buy_vs_benchmark_beat_rate'] = float((benchmark_curve['excess_return'] > 0).mean())
        metrics['buy_vs_benchmark_avg_excess'] = float(benchmark_curve['excess_return'].mean())
    if blend_curve is not None and not blend_curve.empty:
        metrics['buy_vs_blend_beat_rate'] = float((blend_curve['excess_return'] > 0).mean())
        metrics['buy_vs_blend_avg_excess'] = float(blend_curve['excess_return'].mean())

    return metrics


def _build_benchmark_prediction_stats(
    all_recs: pd.DataFrame,
    benchmark_ticker: str,
    buy_threshold: float,
    sell_threshold: float,
) -> Dict[str, Any]:
    """Evaluate model predictions on benchmark ticker itself versus realized outcomes."""
    if all_recs.empty:
        return {}

    bench = all_recs[
        (all_recs['ticker'].str.upper() == benchmark_ticker.upper())
        & all_recs['confidence'].notna()
        & all_recs['perf_1w'].notna()
    ].copy()
    if bench.empty:
        return {}

    bench['date'] = pd.to_datetime(bench['date'])
    bench['pred_dir'] = np.where(
        bench['confidence'] >= buy_threshold,
        'UP',
        np.where(bench['confidence'] <= sell_threshold, 'DOWN', 'NEUTRAL'),
    )
    bench['actual_dir'] = np.where(bench['perf_1w'] > 0, 'UP', 'DOWN')
    bench['predicted_correct_prob'] = np.where(
        bench['pred_dir'] == 'UP',
        bench['confidence'],
        np.where(bench['pred_dir'] == 'DOWN', 1.0 - bench['confidence'], np.nan),
    )
    bench['is_correct'] = bench['pred_dir'] == bench['actual_dir']

    directional = bench[bench['pred_dir'] != 'NEUTRAL'].copy()
    directional_accuracy = float(directional['is_correct'].mean()) if not directional.empty else 0.0
    calibration_error = float(
        np.mean(np.abs(directional['predicted_correct_prob'] - directional['is_correct'].astype(float)))
    ) if not directional.empty else 0.0

    recent_cols = ['date', 'confidence', 'pred_dir', 'actual_dir', 'perf_1w']
    recent = directional.sort_values('date').tail(10)[recent_cols].copy()
    if not recent.empty:
        recent['date'] = recent['date'].dt.strftime('%Y-%m-%d')

    return {
        'samples_total': int(len(bench)),
        'samples_directional': int(len(directional)),
        'directional_accuracy': directional_accuracy,
        'calibration_error': calibration_error,
        'buy_threshold': float(buy_threshold),
        'sell_threshold': float(sell_threshold),
        'recent': recent.to_dict(orient='records') if not recent.empty else [],
    }


def _export_public_model_outputs(
    curve: pd.DataFrame,
    perf_recs: pd.DataFrame,
    benchmark_ticker: str,
    horizon_days: int,
    benchmark_prediction_stats: Dict[str, Any],
    buy_only_metrics: Dict[str, Any],
    sample_flags: Dict[str, Any],
    buy_threshold: float,
    sell_threshold: float,
) -> Optional[Dict[str, str]]:
    """Export public-safe model performance outputs (JSON, Markdown, HTML)."""
    if curve.empty:
        return None

    os.makedirs(PUBLIC_REPORTS_DIR, exist_ok=True)
    bench_token = _sanitize_token(benchmark_ticker)
    json_path = os.path.join(PUBLIC_REPORTS_DIR, f'model_performance_{bench_token}.json')
    md_path = os.path.join(PUBLIC_REPORTS_DIR, f'model_performance_{bench_token}.md')
    html_path = os.path.join(PUBLIC_REPORTS_DIR, f'model_performance_{bench_token}.html')

    strategy_metrics = calculate_metrics(curve['strategy_return'], periods_per_year=52)
    benchmark_metrics = calculate_metrics(curve['benchmark_return'], periods_per_year=52)
    excess_std = curve['excess_return'].std()
    information_ratio = (
        (curve['excess_return'].mean() / excess_std) * np.sqrt(52)
        if excess_std and excess_std > 0
        else 0.0
    )
    beat_rate = float((curve['excess_return'] > 0).mean())
    bucket_df = _build_confidence_bucket_stats(perf_recs)
    directional_metrics = _build_directional_signal_metrics(
        perf_recs=perf_recs,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    monthly_relative = _build_monthly_relative(curve)

    rolling_window = int(min(max(len(curve) // 4, 6), 26))
    rolling = curve.copy()
    rolling['rolling_beat_rate'] = (rolling['excess_return'] > 0).rolling(rolling_window, min_periods=1).mean()
    rolling['rolling_excess_mean'] = rolling['excess_return'].rolling(rolling_window, min_periods=1).mean()
    rolling['strategy_drawdown'] = (rolling['strategy_curve'] / rolling['strategy_curve'].cummax()) - 1.0

    summary_payload = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'benchmark_ticker': benchmark_ticker,
        'horizon_days': int(horizon_days),
        'aligned_dates': int(len(curve)),
        'strategy_total_return': float(curve['strategy_curve'].iloc[-1] - 1.0),
        'benchmark_total_return': float(curve['benchmark_curve'].iloc[-1] - 1.0),
        'total_excess_return': float((curve['strategy_curve'].iloc[-1] / curve['benchmark_curve'].iloc[-1]) - 1.0),
        'beat_rate': beat_rate,
        'information_ratio': float(information_ratio),
        'strategy_metrics': strategy_metrics,
        'benchmark_metrics': benchmark_metrics,
        'confidence_buckets': bucket_df.to_dict(orient='records') if not bucket_df.empty else [],
        'directional_metrics': directional_metrics,
        'buy_only_metrics': buy_only_metrics,
        'sample_adequacy': sample_flags,
        'benchmark_prediction': benchmark_prediction_stats,
        'monthly_relative': monthly_relative.to_dict(orient='records') if not monthly_relative.empty else [],
        'recent_relative': (
            curve.tail(20)[['date', 'strategy_return', 'benchmark_return', 'excess_return']]
            .assign(date=lambda x: pd.to_datetime(x['date']).dt.strftime('%Y-%m-%d'))
            .to_dict(orient='records')
        ),
        'rolling': (
            rolling[['date', 'rolling_beat_rate', 'rolling_excess_mean', 'strategy_drawdown']]
            .assign(date=lambda x: pd.to_datetime(x['date']).dt.strftime('%Y-%m-%d'))
            .to_dict(orient='records')
        ),
        'rolling_window': rolling_window,
    }

    with open(json_path, 'w') as f:
        json.dump(summary_payload, f, indent=2)

    with open(md_path, 'w') as f:
        f.write("# Public Model Performance Summary\n\n")
        f.write(f"Generated: {summary_payload['generated_at']}\n\n")
        f.write(
            f"Model return is equal-weight BUY forward performance over {horizon_days} trading days, "
            f"compared against {benchmark_ticker} over matching dates.\n\n"
        )
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Aligned Dates | {summary_payload['aligned_dates']} |\n")
        f.write(f"| Strategy Total Return | {summary_payload['strategy_total_return']:.2%} |\n")
        f.write(f"| {benchmark_ticker} Total Return | {summary_payload['benchmark_total_return']:.2%} |\n")
        f.write(f"| Total Excess Return | {summary_payload['total_excess_return']:.2%} |\n")
        f.write(f"| Beat Rate | {summary_payload['beat_rate']:.1%} |\n")
        f.write(f"| Information Ratio | {summary_payload['information_ratio']:.2f} |\n\n")

        adequacy = summary_payload.get('sample_adequacy') or {}
        if adequacy and (
            not adequacy.get('directional_sample_ok', False)
            or not adequacy.get('buy_sample_ok', False)
            or not adequacy.get('benchmark_sample_ok', False)
        ):
            f.write("## Reliability Notice\n\n")
            f.write("Interpret these metrics cautiously due to limited sample depth in one or more slices.\n\n")
            f.write("| Slice | Samples | Minimum |\n")
            f.write("|-------|---------|---------|\n")
            f.write(
                f"| Directional outcomes | {adequacy.get('directional_sample_count', 0)} | "
                f"{adequacy.get('thresholds', {}).get('directional_min', 0)} |\n"
            )
            f.write(
                f"| BUY outcomes | {adequacy.get('buy_sample_count', 0)} | "
                f"{adequacy.get('thresholds', {}).get('buy_min', 0)} |\n"
            )
            f.write(
                f"| Benchmark aligned dates | {adequacy.get('benchmark_aligned_dates', 0)} | "
                f"{adequacy.get('thresholds', {}).get('benchmark_dates_min', 0)} |\n"
            )
            f.write("\n")

        if summary_payload['confidence_buckets']:
            f.write("## Confidence Buckets (BUY only)\n\n")
            f.write("| Bucket | Count | Avg Return | Win Rate |\n")
            f.write("|--------|-------|------------|----------|\n")
            for row in summary_payload['confidence_buckets']:
                f.write(
                    f"| {row['bucket']} | {int(row['count'])} | "
                    f"{row['avg_return']:.2%} | {row['win_rate']:.1%} |\n"
                )
            f.write("\n")

        buy_only = summary_payload.get('buy_only_metrics') or {}
        if buy_only:
            f.write("## BUY-only Realized Results\n\n")
            f.write("| Metric | Value |\n|--------|-------|\n")
            f.write(f"| BUY signals with outcomes | {buy_only.get('buy_count', 0)} |\n")
            f.write(f"| Win rate | {buy_only.get('buy_win_rate', 0.0):.1%} |\n")
            f.write(f"| Average return | {buy_only.get('buy_avg_return', 0.0):.2%} |\n")
            f.write(f"| Median return | {buy_only.get('buy_median_return', 0.0):.2%} |\n")
            f.write(f"| Profit factor | {buy_only.get('buy_profit_factor', 0.0):.2f} |\n\n")

        directional = summary_payload.get('directional_metrics') or {}
        if directional:
            f.write("## Directional Call Quality\n\n")
            f.write("| Metric | Value |\n|--------|-------|\n")
            f.write(f"| Total labeled outcomes | {directional.get('sample_count', 0)} |\n")
            f.write(f"| BUY calls | {directional.get('buy_calls', 0)} |\n")
            f.write(f"| SELL calls | {directional.get('sell_calls', 0)} |\n")
            f.write(f"| NEUTRAL calls | {directional.get('neutral_calls', 0)} |\n")
            f.write(f"| BUY win rate | {directional.get('buy_win_rate', 0.0):.1%} |\n")
            f.write(f"| SELL win rate (down moves captured) | {directional.get('sell_win_rate', 0.0):.1%} |\n")
            f.write(f"| Directional accuracy (non-neutral) | {directional.get('directional_accuracy', 0.0):.1%} |\n\n")

        monthly = summary_payload.get('monthly_relative') or []
        if monthly:
            f.write("## Monthly Relative Performance\n\n")
            f.write("| Month | Strategy | Benchmark | Excess |\n")
            f.write("|-------|----------|-----------|--------|\n")
            for row in monthly[-12:]:
                f.write(
                    f"| {row['month']} | {row['strategy_return']:.2%} | "
                    f"{row['benchmark_return']:.2%} | {row['excess_return']:.2%} |\n"
                )
            f.write("\n")

        bench_pred = summary_payload.get('benchmark_prediction') or {}
        if bench_pred:
            f.write(f"## {benchmark_ticker} Prediction vs Actual\n\n")
            f.write(
                f"Directional calls: {bench_pred.get('samples_directional', 0)}/"
                f"{bench_pred.get('samples_total', 0)} | "
                f"Accuracy: {bench_pred.get('directional_accuracy', 0.0):.1%} | "
                f"Calibration Error: {bench_pred.get('calibration_error', 0.0):.3f}\n\n"
            )

        f.write(f"Raw JSON: `reports/public/{os.path.basename(json_path)}`\n")

    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        rolling_df = pd.DataFrame(summary_payload.get('rolling', []))
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Model vs Benchmark Curve",
                "Excess Return by Date",
                f"Rolling Beat Rate ({summary_payload.get('rolling_window', 0)} periods)",
                "Strategy Drawdown",
                "Confidence Bucket Returns",
                f"{benchmark_ticker} Predicted vs Actual",
            ),
            vertical_spacing=0.14,
        )
        fig.add_trace(
            go.Scatter(
                x=curve['date'],
                y=curve['strategy_curve'],
                mode='lines',
                name='Model',
                line=dict(color='#0ea5e9', width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=curve['date'],
                y=curve['benchmark_curve'],
                mode='lines',
                name=benchmark_ticker,
                line=dict(color='#f97316', width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=curve['date'],
                y=curve['excess_return'],
                name='Excess',
                marker_color=np.where(curve['excess_return'] >= 0, '#22c55e', '#ef4444'),
            ),
            row=1,
            col=2,
        )
        if not rolling_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=rolling_df['date'],
                    y=rolling_df['rolling_beat_rate'],
                    mode='lines',
                    name='Rolling Beat Rate',
                    line=dict(color='#0f766e', width=2),
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=rolling_df['date'],
                    y=rolling_df['strategy_drawdown'],
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='#b91c1c', width=2),
                    fill='tozeroy',
                ),
                row=2,
                col=2,
            )

        if not bucket_df.empty:
            fig.add_trace(
                go.Bar(
                    x=bucket_df['bucket'],
                    y=bucket_df['avg_return'],
                    name='Avg Return',
                    marker_color='#6366f1',
                ),
                row=3,
                col=1,
            )

        bench_recent = benchmark_prediction_stats.get('recent', [])
        if bench_recent:
            bench_plot = pd.DataFrame(bench_recent)
            bench_plot['actual_dir_num'] = np.where(bench_plot['actual_dir'] == 'UP', 1, 0)
            fig.add_trace(
                go.Scatter(
                    x=bench_plot['date'],
                    y=bench_plot['confidence'],
                    mode='lines+markers',
                    name='Predicted Prob Up',
                    line=dict(color='#a855f7', width=2),
                ),
                row=3,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=bench_plot['date'],
                    y=bench_plot['actual_dir_num'],
                    mode='markers',
                    name='Actual Up (1/0)',
                    marker=dict(color='#f43f5e', size=8),
                ),
                row=3,
                col=2,
            )

        fig.update_layout(
            title="Model Performance Dashboard (Public)",
            template='plotly_white',
            height=1220,
            legend=dict(orientation='h'),
        )
        fig.update_yaxes(title_text='Cumulative', row=1, col=1)
        fig.update_yaxes(title_text='Excess Return', row=1, col=2)
        fig.update_yaxes(title_text='Beat Rate', range=[0, 1], row=2, col=1)
        fig.update_yaxes(title_text='Drawdown', row=2, col=2)
        fig.update_yaxes(title_text='Avg Return', row=3, col=1)
        fig.update_yaxes(title_text='Probability / Direction', row=3, col=2)
        fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    except Exception as e:
        with open(html_path, 'w') as f:
            f.write("<html><body><h1>Model Performance Dashboard</h1>")
            f.write(f"<p>Dashboard generation fallback: {e}</p>")
            f.write(f"<p>Use JSON at: reports/public/{os.path.basename(json_path)}</p>")
            f.write("</body></html>")

    return {'json_path': json_path, 'md_path': md_path, 'html_path': html_path}


def _get_universe_denyset() -> set:
    """Load configured ticker denylist for cleanup safety."""
    try:
        from config import UNIVERSE_DENYLIST
    except Exception:
        return set()
    return {str(t).upper().strip() for t in (UNIVERSE_DENYLIST or []) if str(t).strip()}


def _purge_denylisted_recommendations(con) -> int:
    """Delete denylisted tickers from recommendation history."""
    denyset = sorted(_get_universe_denyset())
    if not denyset:
        return 0

    placeholders = ",".join(["?"] * len(denyset))
    to_remove = int(
        con.execute(
            f"SELECT COUNT(*) FROM recommendation_history WHERE upper(ticker) IN ({placeholders})",
            denyset,
        ).fetchone()[0]
    )
    if to_remove > 0:
        con.execute(
            f"DELETE FROM recommendation_history WHERE upper(ticker) IN ({placeholders})",
            denyset,
        )
    return to_remove


def _run_open_label_backfill(con) -> Dict[str, int]:
    """Update perf labels for OPEN recommendations and close stale rows."""
    recs = con.execute("SELECT rowid, * FROM recommendation_history WHERE status = 'OPEN'").df()
    stats = {
        'open_rows': int(len(recs)),
        'updated_1w': 0,
        'closed_1m': 0,
        'stale_closed': 0,
        'remaining_stale_open': 0,
    }
    if recs.empty:
        return stats

    today = pd.Timestamp(datetime.now().date())
    for _, rec in recs.iterrows():
        rec_id = rec['rowid']
        ticker = rec['ticker']
        rec_date = rec['date']
        price_at_rec = rec.get('price_at_rec')

        rec_dt = pd.to_datetime(rec_date, errors='coerce')
        if pd.isna(rec_dt):
            continue
        age_days = int((today - rec_dt.normalize()).days)

        try:
            price_at_rec_f = float(price_at_rec)
        except (TypeError, ValueError):
            price_at_rec_f = float('nan')

        if not np.isfinite(price_at_rec_f) or price_at_rec_f <= 0:
            if age_days >= AUDIT_STALE_CLOSE_DAYS:
                con.execute("UPDATE recommendation_history SET status = 'CLOSED' WHERE rowid = ?", [rec_id])
                stats['stale_closed'] += 1
            continue

        prices = con.execute(
            "SELECT date, close FROM price_history WHERE ticker = ? AND date > ? ORDER BY date",
            [ticker, rec_date],
        ).df()

        if prices.empty:
            if age_days >= AUDIT_STALE_CLOSE_DAYS:
                con.execute("UPDATE recommendation_history SET status = 'CLOSED' WHERE rowid = ?", [rec_id])
                stats['stale_closed'] += 1
            continue

        if len(prices) >= 5 and pd.isna(rec.get('perf_1w')):
            price_1w = prices.iloc[4]['close']
            perf_1w = (price_1w / price_at_rec_f) - 1
            con.execute("UPDATE recommendation_history SET perf_1w = ? WHERE rowid = ?", [perf_1w, rec_id])
            stats['updated_1w'] += 1

        if len(prices) >= 21:
            price_1m = prices.iloc[20]['close']
            perf_1m = (price_1m / price_at_rec_f) - 1
            con.execute(
                "UPDATE recommendation_history SET perf_1m = ?, status = 'CLOSED' WHERE rowid = ?",
                [perf_1m, rec_id],
            )
            stats['closed_1m'] += 1
        elif age_days >= AUDIT_STALE_CLOSE_DAYS:
            # Prevent permanently OPEN rows when no month outcome will arrive.
            con.execute("UPDATE recommendation_history SET status = 'CLOSED' WHERE rowid = ?", [rec_id])
            stats['stale_closed'] += 1

    stats['remaining_stale_open'] = int(
        con.execute(
            "SELECT COUNT(*) FROM recommendation_history WHERE status = 'OPEN' AND date <= ?",
            [datetime.now().date() - timedelta(days=AUDIT_STALE_CLOSE_DAYS)],
        ).fetchone()[0]
    )
    return stats


def run_label_backfill() -> Dict[str, int]:
    """Run label backfill/cleanup without full audit report generation."""
    con = get_connection()
    try:
        purged_denylisted = _purge_denylisted_recommendations(con)
        stats = _run_open_label_backfill(con)
        summary = {
            'purged_denylisted': int(purged_denylisted),
            **stats,
        }
        print(
            "[audit] Label backfill: "
            f"open_rows={summary['open_rows']}, "
            f"perf_1w={summary['updated_1w']}, "
            f"closed_1m={summary['closed_1m']}, "
            f"closed_stale={summary['stale_closed']}, "
            f"remaining_stale_open={summary['remaining_stale_open']}",
            flush=True,
        )
        if purged_denylisted > 0:
            print(f"[audit] Purged {purged_denylisted} denylisted recommendation row(s).", flush=True)
        if AUDIT_BLOCK_ON_STALE_OPEN_ROWS and summary['remaining_stale_open'] > AUDIT_MAX_STALE_OPEN_ROWS:
            raise RuntimeError(
                "Audit blocked: stale OPEN rows exceed threshold "
                f"({summary['remaining_stale_open']} > {AUDIT_MAX_STALE_OPEN_ROWS})."
            )
        return summary
    finally:
        con.close()


def run_performance_audit() -> Dict[str, Any]:
    """
    Checks past recommendations, updates performance, and generates a text comparison report.
    """
    con = get_connection()
    dedupe_removed = 0
    purged_denylisted = 0
    backfill_stats = {
        'open_rows': 0,
        'updated_1w': 0,
        'closed_1m': 0,
        'stale_closed': 0,
        'remaining_stale_open': 0,
    }
    try:
        dedupe_removed = int(con.execute("""
            SELECT COUNT(*) FROM (
                SELECT
                    rowid,
                    ROW_NUMBER() OVER (
                        PARTITION BY date, upper(ticker)
                        ORDER BY rowid DESC
                    ) AS rn
                FROM recommendation_history
            ) t
            WHERE rn > 1
        """).fetchone()[0])
        if dedupe_removed > 0:
            con.execute("""
                DELETE FROM recommendation_history
                WHERE rowid IN (
                    SELECT rowid FROM (
                        SELECT
                            rowid,
                            ROW_NUMBER() OVER (
                                PARTITION BY date, upper(ticker)
                                ORDER BY rowid DESC
                            ) AS rn
                        FROM recommendation_history
                    ) t
                    WHERE rn > 1
                )
            """)
            print(f"[audit] Removed {dedupe_removed} duplicate recommendation row(s).")
    except Exception as e:
        print(f"[audit] Warning: could not deduplicate recommendation_history ({e})")

    purged_denylisted = _purge_denylisted_recommendations(con)
    if purged_denylisted > 0:
        print(f"[audit] Purged {purged_denylisted} denylisted recommendation row(s).", flush=True)

    backfill_stats = _run_open_label_backfill(con)
    print(
        f"[audit] OPEN updates: perf_1w={backfill_stats['updated_1w']}, "
        f"closed_1m={backfill_stats['closed_1m']}, "
        f"closed_stale={backfill_stats['stale_closed']}, "
        f"remaining_stale_open={backfill_stats['remaining_stale_open']}",
        flush=True,
    )
    if (
        AUDIT_BLOCK_ON_STALE_OPEN_ROWS
        and backfill_stats['remaining_stale_open'] > AUDIT_MAX_STALE_OPEN_ROWS
    ):
        con.close()
        raise RuntimeError(
            "Audit blocked: stale OPEN rows exceed threshold "
            f"({backfill_stats['remaining_stale_open']} > {AUDIT_MAX_STALE_OPEN_ROWS}). "
            "Run --backfill-labels and verify price history continuity."
        )

    # 2. Generate comparative report from deduped recommendation snapshots.
    all_recs_raw = con.execute("SELECT rowid, * FROM recommendation_history").df()
    con.close()

    if all_recs_raw.empty:
        print("No recommendations found for report.")
        return

    all_recs = (
        all_recs_raw.sort_values('rowid')
        .drop_duplicates(subset=['date', 'ticker'], keep='last')
        .drop(columns=['rowid'])
    )
    raw_count = int(len(all_recs_raw))
    dedup_count = int(len(all_recs))
    if dedup_count != raw_count:
        print(f"[audit] Using {dedup_count} deduped rows from {raw_count} raw recommendations.")

    # Filter for recs with performance data (OPEN or CLOSED).
    perf_recs = all_recs[all_recs['perf_1w'].notna()].copy()
    
    # Even if no perf data, generate a summary report
    print(f"Total recommendations: {len(all_recs)}")
    print(f"With performance data: {len(perf_recs)}")
    
    benchmark_curve = _build_prediction_vs_benchmark_curve(
        perf_recs=perf_recs,
        benchmark_ticker=AUDIT_BENCHMARK_TICKER,
        horizon_days=AUDIT_FORWARD_DAYS,
    )
    benchmark_exports = _export_prediction_vs_benchmark(
        curve=benchmark_curve,
        benchmark_ticker=AUDIT_BENCHMARK_TICKER,
        horizon_days=AUDIT_FORWARD_DAYS,
    )

    if benchmark_exports:
        print(f"Benchmark comparison exported: {benchmark_exports['md_path']}")
        print(f"Benchmark curve CSV: {benchmark_exports['csv_path']}")

    blend_curve = _build_prediction_vs_blend_curve(
        perf_recs=perf_recs,
        horizon_days=AUDIT_FORWARD_DAYS,
        blend_weights=BENCHMARK_BLEND,
    )
    blend_exports = _export_prediction_vs_benchmark(
        curve=blend_curve,
        benchmark_ticker='benchmark_blend',
        horizon_days=AUDIT_FORWARD_DAYS,
    )
    if blend_exports:
        print(f"Blend benchmark exported: {blend_exports['md_path']}")
        print(f"Blend benchmark CSV: {blend_exports['csv_path']}")

    threshold_10d = get_model_thresholds(10)
    active_buy_threshold = float(threshold_10d.get('buy', DEFAULT_BUY_THRESHOLD))
    active_sell_threshold = float(threshold_10d.get('sell', DEFAULT_SELL_THRESHOLD))
    print(f"Using 10d thresholds for audit: BUY>={active_buy_threshold:.2f}, SELL<={active_sell_threshold:.2f}")

    benchmark_pred_stats = _build_benchmark_prediction_stats(
        all_recs=all_recs,
        benchmark_ticker=AUDIT_BENCHMARK_TICKER,
        buy_threshold=active_buy_threshold,
        sell_threshold=active_sell_threshold,
    )
    directional_metrics = _build_directional_signal_metrics(
        perf_recs=perf_recs,
        buy_threshold=active_buy_threshold,
        sell_threshold=active_sell_threshold,
    )
    buy_only_metrics = _build_buy_only_metrics(
        perf_recs=perf_recs,
        benchmark_curve=benchmark_curve,
        blend_curve=blend_curve,
        buy_threshold=active_buy_threshold,
    )
    sample_flags = _build_sample_adequacy_flags(
        directional_metrics=directional_metrics,
        buy_only_metrics=buy_only_metrics,
        benchmark_curve=benchmark_curve,
    )
    public_exports = _export_public_model_outputs(
        curve=benchmark_curve,
        perf_recs=perf_recs,
        benchmark_ticker=AUDIT_BENCHMARK_TICKER,
        horizon_days=AUDIT_FORWARD_DAYS,
        benchmark_prediction_stats=benchmark_pred_stats,
        buy_only_metrics=buy_only_metrics,
        sample_flags=sample_flags,
        buy_threshold=active_buy_threshold,
        sell_threshold=active_sell_threshold,
    )

    if public_exports:
        print(f"Public model summary: {public_exports['md_path']}")
        print(f"Public dashboard: {public_exports['html_path']}")

    # Generate Markdown Report
    report_path = os.path.join(PROJECT_ROOT, 'reports', 'audit_results.md')
    
    try:
        with open(report_path, 'w') as f:
            f.write(f"# 📊 Audit Results: Strategy Performance\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            # Summary counts
            buy_count = len(all_recs[all_recs['signal_type'] == 'BUY'])
            sell_count = len(all_recs[all_recs['signal_type'] == 'SELL'])
            hold_count = len(all_recs[all_recs['signal_type'] == 'HOLD'])
            
            f.write(f"## 1. Recommendation Summary\n\n")
            f.write(f"| Signal | Count | % |\n")
            f.write(f"|--------|-------|----|\n")
            f.write(f"| **BUY** | {buy_count} | {buy_count/max(1,len(all_recs)):.1%} |\n")
            f.write(f"| **SELL** | {sell_count} | {sell_count/max(1,len(all_recs)):.1%} |\n")
            f.write(f"| **HOLD** | {hold_count} | {hold_count/max(1,len(all_recs)):.1%} |\n")
            f.write(f"| **Total** | {len(all_recs)} | 100% |\n\n")

            f.write("### Data Quality\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Raw rows before dedupe | {raw_count} |\n")
            f.write(f"| Rows after dedupe | {dedup_count} |\n")
            f.write(f"| Duplicate rows removed this run | {dedupe_removed} |\n")
            f.write(f"| Denylisted rows purged | {purged_denylisted} |\n")
            f.write(f"| OPEN rows audited | {backfill_stats['open_rows']} |\n")
            f.write(f"| Updated 1W labels | {backfill_stats['updated_1w']} |\n")
            f.write(f"| Closed with 1M labels | {backfill_stats['closed_1m']} |\n")
            f.write(f"| Auto-closed stale OPEN rows | {backfill_stats['stale_closed']} |\n")
            f.write(f"| Remaining stale OPEN rows | {backfill_stats['remaining_stale_open']} |\n\n")

            if (
                not sample_flags.get('directional_sample_ok', False)
                or not sample_flags.get('buy_sample_ok', False)
                or not sample_flags.get('benchmark_sample_ok', False)
            ):
                f.write("### Reliability Notice\n\n")
                f.write("Interpret performance sections cautiously until minimum sample thresholds are met.\n\n")
                f.write("| Slice | Samples | Minimum |\n")
                f.write("|-------|---------|---------|\n")
                f.write(
                    f"| Directional outcomes | {sample_flags.get('directional_sample_count', 0)} | "
                    f"{AUDIT_MIN_DIRECTIONAL_SAMPLES} |\n"
                )
                f.write(
                    f"| BUY outcomes | {sample_flags.get('buy_sample_count', 0)} | "
                    f"{AUDIT_MIN_BUY_SAMPLES} |\n"
                )
                f.write(
                    f"| Benchmark aligned dates | {sample_flags.get('benchmark_aligned_dates', 0)} | "
                    f"{AUDIT_MIN_BENCHMARK_DATES} |\n\n"
                )
            
            if not perf_recs.empty:
                # Calculate stats only if we have performance data
                avg_win_rate = (perf_recs['perf_1w'] > 0).mean()
                avg_return_1w = perf_recs['perf_1w'].mean()
                total_signals = len(perf_recs)
                
                f.write(f"## 2. Strategy Signal Performance\n")
                f.write(f"Based on {total_signals} recommendations with 1-week forward data.\n\n")
                f.write(f"| Metric | Value | Target |\n")
                f.write(f"|--------|-------|--------|\n")
                f.write(f"| **Win Rate (1W)** | {avg_win_rate:.1%} | > 55% |\n")
                f.write(f"| **Avg Return (1W)** | {avg_return_1w:.2%} | > 0.5% |\n\n")
                if not sample_flags.get('directional_sample_ok', False):
                    f.write(
                        f"> ⚠️ Low directional sample depth "
                        f"({sample_flags.get('directional_sample_count', 0)} < {AUDIT_MIN_DIRECTIONAL_SAMPLES}).\n\n"
                    )
                
                f.write(f"## 3. Top Performers (1W)\n")
                top_recs = perf_recs.sort_values('perf_1w', ascending=False).head(10)
                f.write(f"| Date | Ticker | Signal | Return |\n")
                f.write(f"|------|--------|--------|--------|\n")
                for _, r in top_recs.iterrows():
                    f.write(f"| {r['date']} | {r['ticker']} | {r['signal_type']} | **{r['perf_1w']:.2%}** |\n")

                f.write(f"\n## 3B. BUY-only Performance (Realized)\n")
                if not buy_only_metrics:
                    f.write("> No BUY recommendations with forward outcomes yet.\n")
                else:
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    f.write(f"| BUY signals with outcomes | {buy_only_metrics.get('buy_count', 0)} |\n")
                    f.write(f"| BUY win rate | {buy_only_metrics.get('buy_win_rate', 0.0):.1%} |\n")
                    f.write(f"| BUY avg return | {buy_only_metrics.get('buy_avg_return', 0.0):.2%} |\n")
                    f.write(f"| BUY median return | {buy_only_metrics.get('buy_median_return', 0.0):.2%} |\n")
                    f.write(f"| BUY profit factor | {buy_only_metrics.get('buy_profit_factor', 0.0):.2f} |\n")
                    f.write(f"| High-conf BUY count (>= {active_buy_threshold:.0%}) | {buy_only_metrics.get('high_conf_buy_count', 0)} |\n")
                    f.write(f"| High-conf BUY win rate | {buy_only_metrics.get('high_conf_buy_win_rate', 0.0):.1%} |\n")
                    f.write(f"| High-conf BUY avg return | {buy_only_metrics.get('high_conf_buy_avg_return', 0.0):.2%} |\n")
                    if 'buy_vs_benchmark_beat_rate' in buy_only_metrics:
                        f.write(f"| BUY beat rate vs {AUDIT_BENCHMARK_TICKER} | {buy_only_metrics.get('buy_vs_benchmark_beat_rate', 0.0):.1%} |\n")
                        f.write(f"| BUY avg excess vs {AUDIT_BENCHMARK_TICKER} | {buy_only_metrics.get('buy_vs_benchmark_avg_excess', 0.0):.2%} |\n")
                    if 'buy_vs_blend_beat_rate' in buy_only_metrics:
                        f.write(f"| BUY beat rate vs benchmark blend | {buy_only_metrics.get('buy_vs_blend_beat_rate', 0.0):.1%} |\n")
                        f.write(f"| BUY avg excess vs benchmark blend | {buy_only_metrics.get('buy_vs_blend_avg_excess', 0.0):.2%} |\n")
                    if not sample_flags.get('buy_sample_ok', False):
                        f.write(
                            f"\n> ⚠️ BUY-only section has low sample depth "
                            f"({sample_flags.get('buy_sample_count', 0)} < {AUDIT_MIN_BUY_SAMPLES}).\n"
                        )

                f.write(f"\n## 4. Benchmark Comparison ({AUDIT_BENCHMARK_TICKER})\n")
                if benchmark_curve.empty:
                    f.write(f"> Benchmark comparison unavailable: missing overlap between signal dates and {AUDIT_BENCHMARK_TICKER} price history.\n")
                else:
                    strategy_metrics = calculate_metrics(benchmark_curve['strategy_return'], periods_per_year=52)
                    benchmark_metrics = calculate_metrics(benchmark_curve['benchmark_return'], periods_per_year=52)
                    beat_rate = (benchmark_curve['excess_return'] > 0).mean()
                    excess_std = benchmark_curve['excess_return'].std()
                    information_ratio = (
                        (benchmark_curve['excess_return'].mean() / excess_std) * np.sqrt(52)
                        if excess_std and excess_std > 0
                        else 0.0
                    )

                    f.write(
                        f"Based on {len(benchmark_curve)} aligned signal dates using "
                        f"{AUDIT_FORWARD_DAYS}-trading-day forward returns.\n\n"
                    )
                    if not sample_flags.get('benchmark_sample_ok', False):
                        f.write(
                            f"> ⚠️ Benchmark comparison has limited aligned dates "
                            f"({len(benchmark_curve)} < {AUDIT_MIN_BENCHMARK_DATES}).\n\n"
                        )
                    f.write(f"| Metric | Strategy | {AUDIT_BENCHMARK_TICKER} |\n")
                    f.write(f"|--------|----------|-----------|\n")
                    f.write(
                        f"| Total Return | {(benchmark_curve['strategy_curve'].iloc[-1] - 1):.2%} | "
                        f"{(benchmark_curve['benchmark_curve'].iloc[-1] - 1):.2%} |\n"
                    )
                    f.write(
                        f"| CAGR (approx) | {strategy_metrics['cagr']:.2%} | {benchmark_metrics['cagr']:.2%} |\n"
                    )
                    f.write(
                        f"| Sharpe (approx) | {strategy_metrics['sharpe']:.2f} | {benchmark_metrics['sharpe']:.2f} |\n"
                    )
                    f.write(
                        f"| Max Drawdown | {strategy_metrics['max_dd']:.2%} | {benchmark_metrics['max_dd']:.2%} |\n"
                    )
                    f.write(f"| Beat Rate | {beat_rate:.1%} | n/a |\n")
                    f.write(f"| Information Ratio | {information_ratio:.2f} | n/a |\n\n")

                    f.write(f"### Recent Relative Performance\n")
                    f.write(f"| Date | Strategy | {AUDIT_BENCHMARK_TICKER} | Excess |\n")
                    f.write(f"|------|----------|-----------|--------|\n")
                    for _, row in benchmark_curve.tail(10).iterrows():
                        f.write(
                            f"| {pd.to_datetime(row['date']).strftime('%Y-%m-%d')} | "
                            f"{row['strategy_return']:.2%} | {row['benchmark_return']:.2%} | "
                            f"{row['excess_return']:.2%} |\n"
                        )

                    if benchmark_exports:
                        rel_csv = os.path.relpath(benchmark_exports['csv_path'], PROJECT_ROOT)
                        rel_md = os.path.relpath(benchmark_exports['md_path'], PROJECT_ROOT)
                        f.write(
                            f"\nDetailed outputs: `{rel_md}` and `{rel_csv}`.\n"
                        )

                f.write(f"\n## 4B. Benchmark Comparison (Blend)\n")
                if blend_curve.empty:
                    f.write("> Blend comparison unavailable: missing overlap among configured blend components.\n")
                else:
                    blend_strategy_metrics = calculate_metrics(blend_curve['strategy_return'], periods_per_year=52)
                    blend_metrics = calculate_metrics(blend_curve['benchmark_return'], periods_per_year=52)
                    blend_beat_rate = (blend_curve['excess_return'] > 0).mean()
                    blend_excess_std = blend_curve['excess_return'].std()
                    blend_information_ratio = (
                        (blend_curve['excess_return'].mean() / blend_excess_std) * np.sqrt(52)
                        if blend_excess_std and blend_excess_std > 0
                        else 0.0
                    )
                    f.write(
                        f"Based on {len(blend_curve)} aligned signal dates using "
                        f"{AUDIT_FORWARD_DAYS}-trading-day forward returns.\n\n"
                    )
                    if len(blend_curve) < AUDIT_MIN_BENCHMARK_DATES:
                        f.write(
                            f"> ⚠️ Blend comparison has limited aligned dates "
                            f"({len(blend_curve)} < {AUDIT_MIN_BENCHMARK_DATES}).\n\n"
                        )
                    f.write("| Metric | Strategy | Blend |\n")
                    f.write("|--------|----------|-------|\n")
                    f.write(
                        f"| Total Return | {(blend_curve['strategy_curve'].iloc[-1] - 1):.2%} | "
                        f"{(blend_curve['benchmark_curve'].iloc[-1] - 1):.2%} |\n"
                    )
                    f.write(f"| CAGR (approx) | {blend_strategy_metrics['cagr']:.2%} | {blend_metrics['cagr']:.2%} |\n")
                    f.write(f"| Sharpe (approx) | {blend_strategy_metrics['sharpe']:.2f} | {blend_metrics['sharpe']:.2f} |\n")
                    f.write(f"| Max Drawdown | {blend_strategy_metrics['max_dd']:.2%} | {blend_metrics['max_dd']:.2%} |\n")
                    f.write(f"| Beat Rate | {blend_beat_rate:.1%} | n/a |\n")
                    f.write(f"| Information Ratio | {blend_information_ratio:.2f} | n/a |\n\n")
                    if blend_exports:
                        blend_rel_csv = os.path.relpath(blend_exports['csv_path'], PROJECT_ROOT)
                        blend_rel_md = os.path.relpath(blend_exports['md_path'], PROJECT_ROOT)
                        f.write(
                            f"Detailed outputs: `{blend_rel_md}` and `{blend_rel_csv}`.\n"
                        )
            else:
                f.write(f"## 2. Performance Data\n\n")
                f.write(f"> ⚠️ **No 1-week forward performance data available yet.**\n\n")
                f.write(f"Performance data is calculated by comparing recommendation prices to prices 5+ trading days later. ")
                f.write(f"Run `--audit` again after more trading days have elapsed.\n\n")
                
                # Show recent recommendations instead
                f.write(f"## 3. Recent Recommendations\n")
                recent = all_recs.sort_values('date', ascending=False).head(10)
                f.write(f"| Date | Ticker | Signal | Confidence |\n")
                f.write(f"|------|--------|--------|------------|\n")
                for _, r in recent.iterrows():
                    f.write(f"| {r['date']} | {r['ticker']} | {r['signal_type']} | {r['confidence']:.2f} |\n")

        print(f"Audit complete. Report saved to {report_path}")
        return {
            'report_path': report_path,
            'benchmark_exports': benchmark_exports,
            'blend_exports': blend_exports,
            'public_exports': public_exports,
            'benchmark_prediction_stats': benchmark_pred_stats,
            'buy_only_metrics': buy_only_metrics,
            'sample_flags': sample_flags,
            'dedupe_removed': dedupe_removed,
            'purged_denylisted': purged_denylisted,
            'backfill_stats': backfill_stats,
        }
    except Exception as e:
        print(f"Error producing report: {e}")
        import traceback
        traceback.print_exc()
        return {
            'report_path': report_path,
            'benchmark_exports': benchmark_exports,
            'blend_exports': blend_exports,
            'public_exports': public_exports,
            'benchmark_prediction_stats': benchmark_pred_stats,
            'buy_only_metrics': buy_only_metrics,
            'sample_flags': sample_flags,
            'dedupe_removed': dedupe_removed,
            'purged_denylisted': purged_denylisted,
            'backfill_stats': backfill_stats,
        }

def get_signal_leaderboard():
    # ... kept for backward compatibility if needed ...
    pass


if __name__ == "__main__":
    run_performance_audit()
