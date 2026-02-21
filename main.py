import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*urllib3.*')
warnings.filterwarnings('ignore', message='.*NotOpenSSL.*')

import os
import sys
# Prevent __pycache__ generation (helps cloud sync churn).
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True
# Keep src/ imports authoritative; append project root for config resolution.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
import config

import argparse
from reporter import generate_market_report
from scanner import run_full_scan, run_replay_scan
import pandas as pd
from data_loader import split_and_download_ticker
from storage import initialize_db, save_price_data, load_price_data
from signals import generate_signals
from backtest import run_backtest
from audit import run_performance_audit, run_label_backfill, get_signal_leaderboard
from notifier import send_daily_summary_notification, send_weekly_summary_notification

from ml_engine import train_market_model, get_signal_confidence, learn_from_performance

def analyze_ticker(ticker: str):
    print(f"\n--- Analyzing {ticker} ---")
    
    # 1. Update Data
    df_new = split_and_download_ticker(ticker)
    if not df_new.empty:
        save_price_data(df_new)
    
    # 2. Load Data from DB
    df = load_price_data(ticker)
    if df.empty:
        print("No data available.")
        return

    # 3. Generate Signals
    df = generate_signals(df)
    
    # 4. Backtest (Historical Performance)
    stats = run_backtest(df)
    print(f"Historical Performance (5Y):")
    print(f"  CAGR: {stats['cagr']:.2%}")
    print(f"  Max Drawdown: {stats['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    
    # 5. Latest Recommendation
    last_row = df.iloc[-1]
    signal = last_row['signal']
    date = last_row['date']
    
    # ML Confidence
    from ml_engine import extract_ml_features
    df_feat = extract_ml_features(df)
    conf = 0.5
    if not df_feat.empty:
        conf = get_signal_confidence(df_feat.iloc[-1], ticker=ticker)

    rec_text = "WAIT (Cash)"
    if signal == 1:
        rec_text = "Action: HOLD (Maintain Long)"
        if len(df) > 1 and df.iloc[-2]['signal'] == 0:
            rec_text = "Action: BUY (Entry)"
    elif len(df) > 1 and df.iloc[-2]['signal'] == 1:
        rec_text = "Action: SELL (Exit)"
            
    print(f"Latest Signal ({date}): {rec_text}")
    print(f"  Confidence (ML): {conf:.1%}")
    print(f"  Close: {last_row['close']:.2f}")
    print(f"  RSI: {last_row['rsi']:.2f}")
    print(f"  SMA_50: {last_row['sma_50']:.2f}, SMA_200: {last_row['sma_200']:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Stock Market Analyzer')
    parser.add_argument('tickers', nargs='*', help='List of tickers to analyze (e.g. AAPL MSFT)')
    parser.add_argument('--init-db', action='store_true', help='Initialize the database')
    parser.add_argument('--scan', action='store_true', help='Run full market scan')
    parser.add_argument('--replay-scan', action='store_true', help='Replay historical as-of scans and save model_predictions')
    parser.add_argument('--start', type=str, help='Replay start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='Replay end date (YYYY-MM-DD)')
    parser.add_argument('--replay-limit', type=int, help='Optional ticker limit for replay scan')
    parser.add_argument('--audit', action='store_true', help='Run performance audit on past recommendations')
    parser.add_argument('--backfill-labels', action='store_true', help='Backfill OPEN recommendation outcomes and close stale rows')
    parser.add_argument('--train-ml', action='store_true', help='Train the ML meta-classifier on DB history')
    parser.add_argument('--report', action='store_true', help='Generate market report from current DB state')
    parser.add_argument('--learn', action='store_true', help='Learn from past recommendation performance')
    parser.add_argument('--notify', action='store_true', help='Send daily summary notification from latest public model report')
    parser.add_argument('--notify-weekly', action='store_true', help='Send weekly summary notification from latest public model report')
    parser.add_argument('--all', action='store_true', help='[DEPRECATED] Use --pipeline full')
    parser.add_argument('--pipeline', type=str, help='Run steps in order. Comma-separated (scan,backfill,audit,train,report,notify,notify_weekly) or preset (daily, full, daily_auto)')
    
    args = parser.parse_args()
    
    # === PIPELINE EXECUTION ===
    PIPELINE_PRESETS = {
        'daily': ['scan', 'report'],
        'daily_auto': ['scan', 'backfill', 'audit', 'report', 'notify'],
        'weekly_summary': ['backfill', 'audit', 'notify_weekly'],
        'full': ['train', 'scan', 'report'],
        'retrain': ['train'],
        'refresh': ['scan', 'backfill', 'audit', 'report'],
    }
    
    pipeline_steps = None
    if args.pipeline:
        if args.pipeline in PIPELINE_PRESETS:
            pipeline_steps = PIPELINE_PRESETS[args.pipeline]
            print(f"🔄 Using preset pipeline: {args.pipeline} -> {pipeline_steps}")
        else:
            pipeline_steps = [s.strip() for s in args.pipeline.split(',')]
            print(f"🔄 Custom pipeline: {pipeline_steps}")
    elif args.all:
        print("⚠️ --all is deprecated. Use --pipeline full instead.")
        pipeline_steps = PIPELINE_PRESETS['full']
    
    if pipeline_steps:
        print(f"\n🚀 Starting Pipeline: {' → '.join(pipeline_steps)}\n")
        scan_results = None
        scan_meta = None
        audit_result = None
        for i, step in enumerate(pipeline_steps, 1):
            print(f"--- Phase {i}/{len(pipeline_steps)}: {step.upper()} ---")
            try:
                if step == 'scan':
                    scan_results, scan_meta = run_full_scan(return_meta=True)
                elif step in ('replay', 'replay_scan', 'replay-scan'):
                    if not args.start or not args.end:
                        raise ValueError("Replay pipeline step requires --start and --end dates.")
                    run_replay_scan(args.start, args.end, max_tickers=args.replay_limit)
                elif step == 'train':
                    train_market_model()
                elif step == 'learn':
                    learn_from_performance()
                elif step in ('backfill', 'backfill_labels', 'backfill-labels'):
                    run_label_backfill()
                elif step == 'audit':
                    audit_result = run_performance_audit()
                elif step == 'report':
                    generate_market_report(scan_results, scan_meta=scan_meta)
                elif step == 'notify':
                    summary_path = None
                    try:
                        summary_path = ((audit_result or {}).get('public_exports') or {}).get('json_path')
                    except Exception:
                        summary_path = None
                    sent, msg = send_daily_summary_notification(summary_path=summary_path)
                    print(f"  {'✅' if sent else 'ℹ️'} Notification: {msg}")
                elif step in ('notify_weekly', 'notify-weekly'):
                    summary_path = None
                    try:
                        summary_path = ((audit_result or {}).get('public_exports') or {}).get('json_path')
                    except Exception:
                        summary_path = None
                    sent, msg = send_weekly_summary_notification(summary_path=summary_path)
                    print(f"  {'✅' if sent else 'ℹ️'} Weekly Notification: {msg}")
                else:
                    print(f"  ⚠️ Unknown step: {step}. Skipping.")
                print(f"  ✅ {step.upper()} complete.\n")
            except Exception as e:
                print(f"  ❌ {step.upper()} failed: {e}")
                print("  Pipeline halted due to error.")
                return
        print("✅ Pipeline complete.")
        return

    if args.init_db:
        initialize_db()
        return
        
    if args.scan:
        results, scan_meta = run_full_scan(return_meta=True)
        generate_market_report(results, scan_meta=scan_meta)
        return

    if args.replay_scan:
        if not args.start or not args.end:
            print("Replay scan requires --start YYYY-MM-DD and --end YYYY-MM-DD.")
            return
        run_replay_scan(args.start, args.end, max_tickers=args.replay_limit)
        return
        
    if args.report:
        generate_market_report()
        return
        
    if args.learn:
        learn_from_performance()
        return

    if args.notify:
        sent, msg = send_daily_summary_notification()
        print(f"{'✅' if sent else 'ℹ️'} {msg}")
        return

    if args.notify_weekly:
        sent, msg = send_weekly_summary_notification()
        print(f"{'✅' if sent else 'ℹ️'} {msg}")
        return
        
    if args.audit:
        run_performance_audit()
        leaderboard = get_signal_leaderboard()
        print("\n--- Performance Audit Summary ---")
        print(leaderboard)
        return

    if args.backfill_labels:
        run_label_backfill()
        return

    if args.train_ml:
        train_market_model()
        return
        
    if args.tickers:
        initialize_db() # Ensure DB exists
        for ticker in args.tickers:
            analyze_ticker(ticker.upper())

if __name__ == "__main__":
    main()
