import pandas as pd
from datetime import datetime
import os
import json
import math

# Get the absolute path to the project root (market folder)
# Works whether reporter.py is in /market/ or /market/src/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_SCRIPT_DIR) == 'src':
    PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)  # Goes from src/ to market/
else:
    PROJECT_ROOT = _SCRIPT_DIR  # Already in market/
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')


def _to_float(value, default: float = 0.0) -> float:
    """Best-effort numeric conversion for report formatting."""
    try:
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _normalize_dividend_yield(value) -> float:
    """
    Normalize dividend yield to decimal form.
    Some sources may emit percentage points (e.g. 6.5 meaning 6.5%).
    """
    yld = _to_float(value, 0.0)
    if yld < 0:
        return 0.0
    for _ in range(3):
        if yld <= 1.0:
            break
        yld = yld / 100.0
    return max(0.0, yld)


def _to_bool(value, default: bool = False) -> bool:
    """Best-effort bool conversion that treats NA/None safely."""
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    if isinstance(value, str):
        txt = value.strip().lower()
        if txt in {'1', 'true', 'yes', 'y', 'on'}:
            return True
        if txt in {'0', 'false', 'no', 'n', 'off'}:
            return False
        return default
    return bool(value)


def _dedupe_latest_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest row per ticker to avoid duplicate signal rows in reports."""
    if df is None or df.empty or 'ticker' not in df.columns:
        return df
    out = df.copy()
    if 'date' in out.columns:
        out['_report_sort_date'] = pd.to_datetime(out['date'], errors='coerce')
    else:
        out['_report_sort_date'] = pd.NaT
    if 'updated_at' in out.columns:
        out['_report_sort_updated'] = pd.to_datetime(out['updated_at'], errors='coerce')
    else:
        out['_report_sort_updated'] = pd.NaT
    out = out.sort_values(
        by=['ticker', '_report_sort_date', '_report_sort_updated'],
        ascending=[True, True, True],
        kind='mergesort',
    )
    out = out.drop_duplicates(subset=['ticker'], keep='last')
    return out.drop(columns=['_report_sort_date', '_report_sort_updated'], errors='ignore')

def _get_benchmark_yield(df: pd.DataFrame) -> float:
    """Return benchmark yield for safe-asset gating (fixed 7-day yield)."""
    from config import SAFE_ASSET_BENCHMARK_YIELD_7D
    return float(SAFE_ASSET_BENCHMARK_YIELD_7D or 0.0)

def _get_term_str(row) -> str:
    """Format Term Structure with emoji indicators (matches SCAN_STATUS)."""
    c5 = row.get('conf_5d', 0.5) if isinstance(row, dict) else row.get('conf_5d', 0.5)
    c10 = row.get('confidence', 0.5) if isinstance(row, dict) else row.get('confidence', 0.5)
    c30 = row.get('conf_30d', 0.5) if isinstance(row, dict) else row.get('conf_30d', 0.5)
    
    def fmt(c):
        if c == 0:
            return "N/A"
        emoji = "🟢" if c >= 0.6 else ("🔴" if c <= 0.4 else "➖")
        return f"{emoji} {c:.0%}"
    
    return f"{fmt(c5)}|{fmt(c10)}|{fmt(c30)}"

def _get_tech_summary(row) -> str:
    """Format Technicals summary (RSI + Trend vs SMA200)."""
    rsi_val = row.get('rsi', 0) if isinstance(row, dict) else row.get('rsi', 0)
    price = row.get('price_at_rec', 0) if isinstance(row, dict) else row.get('price_at_rec', 0)
    sma = row.get('sma_200', 0) if isinstance(row, dict) else row.get('sma_200', 0)
    
    rsi_str = f"{rsi_val:.0f}" if pd.notna(rsi_val) else "N/A"
    trend_sym = '↑' if price > sma else '⬇'
    return f"RSI:{rsi_str} {trend_sym} SMA"

def _get_safety_rating(vol: float) -> str:
    """Get safety star rating based on volatility (ATR ratio)."""
    if vol < 0.0005:
        return "⭐⭐⭐"
    elif vol < 0.0015:
        return "⭐⭐"
    elif vol < 0.0025:
        return "⭐"
    else:
        return "⚠️"

def export_ai_readable_data(df: pd.DataFrame, date_str: str) -> str:
    """
    Exports COMPACT market data in JSON for AI analysis.
    Includes top BUY/SELL signals sorted by confidence with term structure.
    """
    ai_file = os.path.join(REPORTS_DIR, f'ai_data_{date_str}.json')

    def build_record(row: pd.Series) -> dict:
        return {
            't': row['ticker'],  # Ticker
            's': row['signal_type'][0],  # B or S
            'p': round(float(row['price_at_rec']), 2),
            'c10': round(float(row.get('confidence', 0.5)), 3),
            'c5': round(float(row.get('conf_5d', 0.5)), 3),
            'c30': round(float(row.get('conf_30d', 0.5)), 3),
            'rsi': round(float(row.get('rsi', 0) or 0), 1),
            'trend': 1 if row['price_at_rec'] > row['sma_200'] else 0,  # 1=bullish
            'div': 1 if _to_bool(row.get('is_dividend'), False) else 0,
            'trad': 1 if _to_bool(row.get('tradable', True), True) else 0,
            'yld': round(_normalize_dividend_yield(row.get('dividend_yield', 0) or 0), 4),
            'ret30': round(float(row.get('returns_30d', 0) or 0), 4),
        }

    actionable = df[(df['signal_type'] == 'BUY') | (df['signal_type'] == 'SELL')]
    if 'tradable' in actionable.columns:
        tradable_mask = actionable['tradable'].map(lambda v: _to_bool(v, True))
        actionable = actionable[
            (actionable['signal_type'] != 'BUY')
            | tradable_mask
        ]
    actionable_count = len(actionable)
    top_buys = actionable[actionable['signal_type'] == 'BUY'].sort_values('confidence', ascending=False).head(100)
    top_sells = actionable[actionable['signal_type'] == 'SELL'].sort_values('confidence', ascending=True).head(100)

    output = {
        'ts': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'legend': {
            't': 'ticker',
            's': 'signal (B=BUY,S=SELL)',
            'p': 'price_at_rec',
            'c10': '10d ML confidence',
            'c5': '5d ML confidence',
            'c30': '30d ML confidence',
            'rsi': 'RSI (reporting only)',
            'trend': 'price_above_sma200 (1=yes)',
            'div': 'dividend flag (1=yes)',
            'trad': 'tradable/liquid filter pass (1=yes)',
            'yld': 'dividend_yield (decimal)',
            'ret30': '30d return (decimal)',
        },
        'n': len(df),
        'buys': len(df[df['signal_type'] == 'BUY']),
        'sells': len(df[df['signal_type'] == 'SELL']),
        'avg_c10': round(float(df['confidence'].mean()), 3) if 'confidence' in df.columns else 0.5,
        'top_buys': [build_record(row) for _, row in top_buys.iterrows()],
        'top_sells': [build_record(row) for _, row in top_sells.iterrows()],
    }
    
    with open(ai_file, 'w') as f:
        json.dump(output, f, separators=(',', ':'))  # No indentation for compactness
    
    print(f"AI-readable data exported: {ai_file} ({actionable_count} actionable signals)")
    return ai_file

def generate_market_report(results: list = None, scan_meta: dict = None):
    """Generates markdown report aligned with SCAN_STATUS format."""
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        
    date_str = datetime.now().strftime('%Y-%m-%d')
    file_path = os.path.join(REPORTS_DIR, f'market_report_{date_str}.md')
    
    if results is None or len(results) == 0:
        from storage import get_connection
        con = get_connection()
        today = datetime.now().strftime('%Y-%m-%d')
        # Use the latest scan for today (preserves multiple intraday scans)
        df = con.execute("""
            SELECT * FROM recommendation_history
            WHERE (date = ? AND scan_time = (
                SELECT MAX(scan_time) FROM recommendation_history WHERE date = ?
            ))
            OR (date = ? AND scan_time IS NULL)
            OR status = 'OPEN'
        """, [today, today, today]).df()
        con.close()
    else:
        df = pd.DataFrame(results)

    df = _dedupe_latest_by_ticker(df)

    if df.empty:
        print("No results to report.")
        return

    # Load config
    from config import (
        SAFE_ASSET_ATR_THRESHOLD,
        SAFE_ASSET_ALLOWLIST,
        SAFE_ASSET_DENYLIST,
        SAFE_ASSET_MIN_YIELD_MULTIPLIER,
        SAFE_ASSET_RETURN_DAYS,
        SAFE_ASSET_MIN_RETURN_ABS,
        SAFE_ASSET_BENCHMARK_YIELD_7D,
        PORTFOLIO_HOLDINGS,
        WATCHLIST,
        UNIVERSE_DENYLIST,
        INCLUDE_HOLDINGS_REPORT,
        INCLUDE_WATCHLIST_REPORT,
    )

    universe_denyset = {t.upper() for t in UNIVERSE_DENYLIST}
    if universe_denyset:
        df = df[~df['ticker'].str.upper().isin(universe_denyset)].copy()
        if df.empty:
            print("No results to report after universe denylist filtering.")
            return

    # Normalize dividend yield for reporting/AI payload consistency.
    if 'dividend_yield' in df.columns:
        df['dividend_yield_norm'] = df['dividend_yield'].apply(_normalize_dividend_yield)
    else:
        df['dividend_yield_norm'] = 0.0

    # Export AI-readable JSON
    export_ai_readable_data(df, date_str)
    
    benchmark_yield = _get_benchmark_yield(df)
    
    # Identify Safe Assets
    allowlist = {t.upper() for t in SAFE_ASSET_ALLOWLIST}
    denylist = {t.upper() for t in SAFE_ASSET_DENYLIST}
    tickers_upper = df['ticker'].str.upper()

    allow_mask = tickers_upper.isin(allowlist) if allowlist else True
    deny_mask = tickers_upper.isin(denylist)

    atr_vals = df['atr_ratio'].fillna(0) if 'atr_ratio' in df.columns else pd.Series(0, index=df.index)
    div_vals = df['dividend_yield_norm'].fillna(0)
    ret_vals = df['returns_30d'].fillna(float('nan')) if 'returns_30d' in df.columns else pd.Series(float('nan'), index=df.index)
    min_yield = benchmark_yield * SAFE_ASSET_MIN_YIELD_MULTIPLIER
    min_return = SAFE_ASSET_MIN_RETURN_ABS
    total_return = ret_vals + (div_vals * (SAFE_ASSET_RETURN_DAYS / 365.0))
    benchmark_total = benchmark_yield * (SAFE_ASSET_RETURN_DAYS / 365.0) * SAFE_ASSET_MIN_YIELD_MULTIPLIER

    use_existing_safe = 'is_safe_asset' in df.columns and df['is_safe_asset'].notna().any()
    if not use_existing_safe:
        df['is_safe_asset'] = (
            allow_mask
            & ~deny_mask
            & (atr_vals > 0)
            & (atr_vals < SAFE_ASSET_ATR_THRESHOLD)
            & (div_vals >= min_yield)
            & (ret_vals >= min_return)
            & (total_return >= benchmark_total)
        )

    # Create sets for categorization
    holdings_set = {h.upper() for h in PORTFOLIO_HOLDINGS}
    watchlist_set = {w.upper() for w in WATCHLIST}
    
    # Categorize results (like SCAN_STATUS does)
    safe_assets = df[df['is_safe_asset'] == True].sort_values('dividend_yield_norm', ascending=False)
    portfolio_df = df[df['ticker'].str.upper().isin(holdings_set)].drop_duplicates(subset=['ticker'], keep='last').sort_values('confidence', ascending=False)
    watchlist_df = df[df['ticker'].str.upper().isin(watchlist_set)]
    
    # BUY/SELL signals (exclude portfolio, watchlist, safe assets)
    excluded_tickers = holdings_set | watchlist_set | set(safe_assets['ticker'].str.upper())
    main_df = df[~df['ticker'].str.upper().isin(excluded_tickers)]
    
    buy_mask = main_df['signal_type'] == 'BUY'
    if 'tradable' in main_df.columns:
        tradable_buy_mask = main_df['tradable'].map(lambda v: _to_bool(v, True))
        buy_mask = buy_mask & tradable_buy_mask
    buys = main_df[buy_mask].sort_values('confidence', ascending=False)
    sells = main_df[main_df['signal_type'] == 'SELL'].sort_values('confidence', ascending=True)
    
    # Market stats
    total_scanned = len(df)
    above_200 = len(df[df['price_at_rec'] > df['sma_200']])
    breadth = above_200 / total_scanned if total_scanned > 0 else 0
    avg_conf = df['confidence'].mean() if 'confidence' in df.columns else 0.5
    
    stale_from_df = 0
    if 'data_stale' in df.columns:
        stale_from_df = int(df['data_stale'].fillna(False).astype(bool).sum())

    meta = scan_meta or {}
    total_tickers_meta = int(meta.get('total_tickers', total_scanned) or total_scanned)
    results_generated_meta = int(meta.get('results_generated', total_scanned) or total_scanned)
    stale_tickers_meta = int(meta.get('stale_tickers', stale_from_df) or stale_from_df)
    failed_fetch_meta = int(meta.get('failed_fetch_tickers', max(total_tickers_meta - results_generated_meta, 0)) or 0)
    insufficient_history_meta = int(meta.get('insufficient_history_tickers', 0) or 0)
    healthy_fetched = max(results_generated_meta - stale_tickers_meta, 0)
    coverage = (results_generated_meta / total_tickers_meta) if total_tickers_meta > 0 else 0.0
    
    # Macro regime
    try:
        from macro_loader import get_macro_regime
        macro_regime = get_macro_regime()
        regime_emoji = '🟢' if macro_regime == 'RISK_ON' else ('🔴' if macro_regime == 'RISK_OFF' else '⚪')
    except Exception:
        macro_regime = 'N/A'
        regime_emoji = '⚪'

    # Agent-first snapshot (stable markers for downstream parser).
    exit_alerts = []
    if not portfolio_df.empty:
        exit_alerts = portfolio_df[portfolio_df['signal_type'] == 'SELL']['ticker'].tolist()

    def _row_snapshot(row, include_tech=False):
        payload = {
            "ticker": str(row.get('ticker', '')),
            "price": round(_to_float(row.get('price_at_rec', 0.0), 0.0), 2),
            "conf_5d": round(_to_float(row.get('conf_5d', 0.5), 0.5), 3),
            "conf_10d": round(_to_float(row.get('confidence', 0.5), 0.5), 3),
            "conf_30d": round(_to_float(row.get('conf_30d', 0.5), 0.5), 3),
            "dividend": _to_bool(row.get('is_dividend', False), False),
        }
        if include_tech:
            payload["rsi"] = round(_to_float(row.get('rsi', 0.0), 0.0), 1)
            payload["above_sma_200"] = bool(_to_float(row.get('price_at_rec', 0.0), 0.0) > _to_float(row.get('sma_200', 0.0), 0.0))
        return payload

    top_buys_payload = [_row_snapshot(row, include_tech=True) for _, row in buys.head(10).iterrows()]
    top_sells_payload = [_row_snapshot(row, include_tech=True) for _, row in sells.head(10).iterrows()]

    net_signal = len(buys) - len(sells)
    if net_signal >= 15:
        signal_bias = "bullish"
    elif net_signal <= -15:
        signal_bias = "defensive"
    else:
        signal_bias = "balanced"

    actions = []
    if exit_alerts:
        actions.append(f"review_exit_alerts:{','.join(exit_alerts)}")
    if len(buys) > 0:
        actions.append("review_top_buys")
    if failed_fetch_meta > 0 or stale_tickers_meta > 0:
        actions.append("check_data_health")
    if not actions:
        actions.append("monitor_no_urgent_actions")

    agent_payload = {
        "schema_version": "market_report_agent_v1",
        "report_date": date_str,
        "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M'),
        "market": {
            "macro_regime": macro_regime,
            "macro_regime_emoji": regime_emoji,
            "signal_bias": signal_bias,
            "avg_confidence_10d": round(_to_float(avg_conf, 0.5), 4),
            "breadth_above_sma200": round(_to_float(breadth, 0.0), 4),
        },
        "counts": {
            "tickers_scanned": int(total_scanned),
            "buys": int(len(buys)),
            "sells": int(len(sells)),
            "safe_assets": int(len(safe_assets)),
            "portfolio": int(len(portfolio_df)),
            "watchlist": int(len(watchlist_df)),
        },
        "data_health": {
            "coverage": round(_to_float(coverage, 0.0), 4),
            "fetched": int(results_generated_meta),
            "target_universe": int(total_tickers_meta),
            "fresh": int(healthy_fetched),
            "stale": int(stale_tickers_meta),
            "fetch_failures": int(failed_fetch_meta),
            "insufficient_history": int(insufficient_history_meta),
        },
        "alerts": {
            "exit_alerts": exit_alerts,
        },
        "top_buys": top_buys_payload,
        "top_sells": top_sells_payload,
        "actions": actions,
        "ai_data_file": f"reports/ai_data_{date_str}.json",
    }
    
    with open(file_path, 'w') as f:
        f.write(f"# Market Analysis Report - {date_str}\n\n")
        
        # Signal Legend (like SCAN_STATUS)
        f.write(f"### ℹ️ Signal Legend\n")
        f.write(f"> * **Term Structure**: 5d|10d|30d ML Confidence (🟢>60%, 🔴<40%).\n")
        f.write(f"> * **Technicals**: RSI (Relative Strength Index) | Trend against SMA200 (↑/⬇).\n")
        f.write(f"> * **Safe Assets**: Allowlist + ATR<{SAFE_ASSET_ATR_THRESHOLD:.1%} + Yield >= {SAFE_ASSET_BENCHMARK_YIELD_7D:.2%} (7d benchmark) + {SAFE_ASSET_RETURN_DAYS}d Return >= {SAFE_ASSET_MIN_RETURN_ABS:.2%}.\n")
        f.write(f"> * **Div**: 💰 indicates dividend paying stock.\n\n")

        f.write("## Agent Morning Snapshot\n")
        f.write("<!-- OPENCLAW:SUMMARY:START -->\n")
        f.write(f"- report_date: {date_str}\n")
        f.write(f"- generated_at: {agent_payload['generated_at']}\n")
        f.write(f"- regime: {regime_emoji} {macro_regime}\n")
        f.write(f"- signal_bias: {signal_bias}\n")
        f.write(f"- buys: {len(buys)}\n")
        f.write(f"- sells: {len(sells)}\n")
        f.write(f"- exit_alerts: {','.join(exit_alerts) if exit_alerts else 'none'}\n")
        f.write(f"- data_health: coverage={coverage:.1%}, stale={stale_tickers_meta}, fetch_failures={failed_fetch_meta}\n")
        f.write(f"- top_buys: {','.join([x['ticker'] for x in top_buys_payload[:5]]) if top_buys_payload else 'none'}\n")
        f.write(f"- top_sells: {','.join([x['ticker'] for x in top_sells_payload[:5]]) if top_sells_payload else 'none'}\n")
        f.write(f"- actions: {','.join(actions)}\n")
        f.write("<!-- OPENCLAW:SUMMARY:END -->\n\n")

        f.write("### Agent Payload (JSON)\n")
        f.write("<!-- OPENCLAW:JSON:START -->\n")
        f.write("```json\n")
        f.write(json.dumps(agent_payload, indent=2))
        f.write("\n```\n")
        f.write("<!-- OPENCLAW:JSON:END -->\n\n")
        
        # Market Summary
        f.write(f"## Market Summary\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Tickers Scanned | {total_scanned} |\n")
        f.write(f"| Above 200D SMA | {breadth:.1%} ({above_200}/{total_scanned}) |\n")
        f.write(f"| Macro Regime | {regime_emoji} {macro_regime} |\n")
        f.write(f"| Avg ML Confidence | {avg_conf:.1%} |\n\n")

        f.write("### Data Health\n")
        f.write(f"- Universe tickers (target): {total_tickers_meta}\n")
        f.write(f"- Tickers fetched/analyzed: {results_generated_meta} ({coverage:.1%} coverage)\n")
        f.write(f"- Fresh data tickers: {healthy_fetched}\n")
        f.write(f"- Stale data tickers: {stale_tickers_meta}\n")
        f.write(f"- Fetch failures (no data returned): {failed_fetch_meta}\n")
        f.write(f"- Insufficient history (<200 bars): {insufficient_history_meta}\n\n")
        
        # Statistics (like SCAN_STATUS)
        f.write(f"### Statistics\n")
        f.write(f"- 🟢 Buy Candidates: {len(buys)}\n")
        f.write(f"- 🛡️ Safe Assets: {len(safe_assets)}\n")
        f.write(f"- 💼 Portfolio: {len(portfolio_df)}\n")
        f.write(f"- 🔴 Sells: {len(sells)}\n\n")
        
        # 1. Portfolio Holdings (FIRST - like SCAN_STATUS)
        if INCLUDE_HOLDINGS_REPORT and not portfolio_df.empty:
            f.write(f"## 💼 Portfolio Holdings Status\n")
            f.write("| Ticker | Price | Term Structure (5d|10d|30d) | Technicals | Div |\n")
            f.write("|--------|-------|-----------------------------|------------|-----|\n")
            for _, row in portfolio_df.iterrows():
                div = '💰' if _to_bool(row.get('is_dividend'), False) else ''
                f.write(f"| {row['ticker']} | {row['price_at_rec']:.2f} | {_get_term_str(row)} | {_get_tech_summary(row)} | {div} |\n")
            
            # Alert for SELL signals on holdings
            sell_alerts = portfolio_df[portfolio_df['signal_type'] == 'SELL']
            if not sell_alerts.empty:
                f.write(f"\n> ⚠️ **EXIT ALERT**: {', '.join(sell_alerts['ticker'].tolist())} have SELL signals!\n")
            f.write("\n")
        
        # 2. Buy Opportunities
        f.write(f"## 🟢 Top Buy Opportunities ({len(buys)} total)\n")
        if not buys.empty:
            f.write("| Ticker | Price | Term Structure (5d|10d|30d) | Technicals | Div |\n")
            f.write("|--------|-------|-----------------------------|------------|-----|\n")
            for _, row in buys.head(15).iterrows():
                div = '💰' if _to_bool(row.get('is_dividend'), False) else ''
                f.write(f"| {row['ticker']} | {row['price_at_rec']:.2f} | {_get_term_str(row)} | {_get_tech_summary(row)} | {div} |\n")
        else:
            f.write("No BUY signals.\n")
        f.write("\n")
        
        # 3. Watchlist
        if INCLUDE_WATCHLIST_REPORT and not watchlist_df.empty:
            f.write(f"## 👀 Watchlist Status\n")
            f.write("| Ticker | Price | Term Structure (5d|10d|30d) | Div |\n")
            f.write("|--------|-------|-----------------------------|-----|\n")
            for _, row in watchlist_df.sort_values('confidence', ascending=False).iterrows():
                div = '💰' if _to_bool(row.get('is_dividend'), False) else ''
                f.write(f"| {row['ticker']} | {row['price_at_rec']:.2f} | {_get_term_str(row)} | {div} |\n")
            f.write("\n")
        
        # 4. Safe Assets
        if not safe_assets.empty:
            f.write(f"## 🛡️ Top Safe Assets (Yield/Income)\n")
            f.write(f"> Benchmark 7d Yield: {SAFE_ASSET_BENCHMARK_YIELD_7D:.2%}\n")
            f.write("| Ticker | Price | Yield | 30d Ret | Spread | Vol | Safety |\n")
            f.write("|--------|-------|-------|---------|--------|-----|--------|\n")
            for _, row in safe_assets.head(15).iterrows():
                yld = _normalize_dividend_yield(row.get('dividend_yield_norm', row.get('dividend_yield', 0)) or 0)
                yld_val = f"{yld*100:.2f}%"
                ret_30d = row.get('returns_30d')
                ret_val = f"{ret_30d:.2%}" if pd.notna(ret_30d) else "N/A"
                spread = yld - SAFE_ASSET_BENCHMARK_YIELD_7D
                spread_val = f"+{spread*100:.1f}%" if spread >= 0 else f"{spread*100:.1f}%"
                vol = row.get('atr_ratio', 0) or 0
                vol_val = f"{vol:.2%}"
                safety = _get_safety_rating(vol)
                f.write(f"| {row['ticker']} | {row['price_at_rec']:.2f} | {yld_val} | {ret_val} | {spread_val} | {vol_val} | {safety} |\n")
            f.write("\n")
        
        # 5. Sell Signals
        f.write(f"## 🔴 Latest SELL Signals ({len(sells)} total)\n")
        if not sells.empty:
            f.write("| Ticker | Price | Term Structure (5d|10d|30d) | Technicals |\n")
            f.write("|--------|-------|-----------------------------|------------|\n")
            for _, row in sells.head(15).iterrows():
                f.write(f"| {row['ticker']} | {row['price_at_rec']:.2f} | {_get_term_str(row)} | {_get_tech_summary(row)} |\n")
        else:
            f.write("No SELL signals.\n")
        f.write("\n")
        
        # Sector/Country Analysis (optional, condensed)
        try:
            from sector_country import aggregate_by_sector, aggregate_by_country, get_sector_heatmap_md, get_country_summary_md
            sector_agg = aggregate_by_sector(results if results else df.to_dict('records'))
            country_agg = aggregate_by_country(results if results else df.to_dict('records'))
            f.write(get_sector_heatmap_md(sector_agg) + "\n\n")
            f.write(get_country_summary_md(country_agg) + "\n\n")
        except Exception as e:
            pass  # Skip if unavailable
        
        f.write(f"---\n*AI data available at: `reports/ai_data_{date_str}.json`*\n")

    print(f"Report generated: {file_path}")
    return file_path
