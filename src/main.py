import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*urllib3.*')
warnings.filterwarnings('ignore', message='.*NotOpenSSL.*')

import os
import sys
import subprocess
# Prevent __pycache__ generation (helps cloud sync churn).
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True
# Keep src/ imports authoritative; append project root for config resolution.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
import config

import argparse
import json
from reporter import generate_market_report
from scanner import run_full_scan, run_replay_scan, build_calibration_artifacts
import pandas as pd
from data_loader import split_and_download_ticker
from storage import initialize_db, save_price_data, load_price_data
from signals import generate_signals
from backtest import run_backtest
from audit import run_performance_audit, run_label_backfill, get_signal_leaderboard
from notifier import send_daily_summary_notification, send_weekly_summary_notification

from ml_engine import train_market_model, get_signal_confidence, learn_from_performance


def _parse_horizons_arg(raw: str):
    """Parse comma-separated horizon list into sorted integers."""
    if not raw:
        return None
    values = []
    for token in str(raw).split(','):
        token = token.strip()
        if not token:
            continue
        try:
            val = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid horizon token '{token}'. Use comma-separated integers.") from exc
        if val <= 0:
            raise ValueError(f"Invalid horizon value '{val}'. Horizons must be positive integers.")
        values.append(val)
    if not values:
        return None
    return sorted(set(values))


def _verify_model_artifacts(runtime_skew_output: str = '') -> str:
    """
    Build artifact manifest then run strict runtime skew policy checks.

    Returns:
        Absolute path to runtime skew JSON summary.
    """
    reports_dir = os.path.join(PROJECT_ROOT, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    if runtime_skew_output:
        output_path = os.path.abspath(runtime_skew_output)
    else:
        ts = pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(reports_dir, f'runtime_skew_{ts}.json')

    manifest_script = os.path.join(PROJECT_ROOT, 'scripts', 'build_model_manifest.py')
    skew_script = os.path.join(PROJECT_ROOT, 'scripts', 'check_model_runtime_skew.py')

    build_cmd = [sys.executable, manifest_script]
    build_proc = subprocess.run(build_cmd, capture_output=True, text=True)
    if build_proc.returncode != 0:
        detail = (build_proc.stderr or build_proc.stdout or '').strip()
        raise RuntimeError(f"Manifest build failed: {detail}")

    skew_cmd = [sys.executable, skew_script, '--strict', '--output', output_path]
    skew_proc = subprocess.run(skew_cmd, capture_output=True, text=True)
    if skew_proc.stdout:
        lines = [ln for ln in skew_proc.stdout.strip().splitlines() if ln.strip()]
        if lines:
            try:
                summary = json.loads(lines[-1])
                policy = summary.get('policy') or {}
                print(
                    "  Artifact policy check: "
                    f"passes={bool(policy.get('passes', False))} "
                    f"violations={int(policy.get('violation_count', 0))}"
                )
            except Exception:
                pass
    if skew_proc.returncode != 0:
        detail = (skew_proc.stderr or '').strip()
        raise RuntimeError(
            "Artifact policy check failed in strict mode. "
            f"runtime_skew_report={output_path} detail={detail}"
        )
    print(f"  Runtime skew report: {output_path}")
    return output_path


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
    parser.add_argument('--build-calibration-artifacts', action='store_true', help='Build calibration JSON artifacts from DB replay predictions (no live API fetch).')
    parser.add_argument('--start', type=str, help='Replay/calibration start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='Replay/calibration end date (YYYY-MM-DD)')
    parser.add_argument('--replay-limit', type=int, help='Optional ticker limit for replay scan')
    parser.add_argument('--verify-artifacts', action='store_true', help='Build model manifest and run strict artifact policy checks.')
    parser.add_argument('--runtime-skew-output', type=str, help='Optional output path for runtime skew JSON report when running artifact verification.')
    parser.add_argument('--calibration-horizons', type=str, help='Optional comma-separated horizons for calibration build (default: config HORIZONS)')
    parser.add_argument('--calibration-window-days', type=int, default=365, help='Default lookback days when calibration start/end dates are omitted')
    parser.add_argument('--no-replay-refresh', action='store_true', help='Do not auto-run replay to backfill missing calibration input rows')
    parser.add_argument('--audit', action='store_true', help='Run performance audit on past recommendations')
    parser.add_argument('--backfill-labels', action='store_true', help='Backfill OPEN recommendation outcomes and close stale rows')
    parser.add_argument('--train-ml', action='store_true', help='Train the ML meta-classifier on DB history')
    parser.add_argument('--report', action='store_true', help='Generate market report from current DB state')
    parser.add_argument('--learn', action='store_true', help='Learn from past recommendation performance')
    parser.add_argument('--notify', action='store_true', help='Send daily summary notification from latest public model report')
    parser.add_argument('--notify-weekly', action='store_true', help='Send weekly summary notification from latest public model report')
    parser.add_argument('--update-news', action='store_true', help='Fetch daily market news assessment via Gemini CLI')
    parser.add_argument('--all', action='store_true', help='[DEPRECATED] Use --pipeline full')
    parser.add_argument('--pipeline', type=str, help='Run steps in order. Comma-separated (scan,backfill,audit,train,calibrate,verify_artifacts,report,notify,notify_weekly,update_news) or preset (daily, full, daily_auto)')
    
    args = parser.parse_args()
    
    # === PIPELINE EXECUTION ===
    PIPELINE_PRESETS = {
        'daily': ['update_news', 'scan', 'report'],
        'daily_auto': ['update_news', 'scan', 'backfill', 'audit', 'report', 'notify'],
        'weekly_summary': ['backfill', 'audit', 'notify_weekly'],
        'full': ['update_news', 'train', 'calibrate', 'verify_artifacts', 'scan', 'report'],
        'verify': ['verify_artifacts'],
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
                if step in ('update_news', 'update-news', 'news'):
                    from news_loader import fetch_daily_news
                    fetch_daily_news()
                elif step == 'scan':
                    scan_results, scan_meta = run_full_scan(return_meta=True)
                elif step in ('replay', 'replay_scan', 'replay-scan'):
                    if not args.start or not args.end:
                        raise ValueError("Replay pipeline step requires --start and --end dates.")
                    run_replay_scan(args.start, args.end, max_tickers=args.replay_limit)
                elif step in ('calibrate', 'calibration', 'build_calibration_artifacts', 'build-calibration-artifacts'):
                    horizons = _parse_horizons_arg(args.calibration_horizons)
                    build_calibration_artifacts(
                        start_date=args.start,
                        end_date=args.end,
                        horizons=horizons,
                        max_tickers=args.replay_limit,
                        replay_if_missing=(not args.no_replay_refresh),
                        window_days=args.calibration_window_days,
                    )
                elif step in ('verify_artifacts', 'verify-artifacts', 'verify', 'artifact_policy', 'artifact-policy'):
                    _verify_model_artifacts(runtime_skew_output=args.runtime_skew_output or '')
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

    if args.verify_artifacts:
        try:
            _verify_model_artifacts(runtime_skew_output=args.runtime_skew_output or '')
        except Exception as e:
            print(f"Artifact verification failed: {e}")
            return
        return

    if args.build_calibration_artifacts:
        try:
            horizons = _parse_horizons_arg(args.calibration_horizons)
            build_calibration_artifacts(
                start_date=args.start,
                end_date=args.end,
                horizons=horizons,
                max_tickers=args.replay_limit,
                replay_if_missing=(not args.no_replay_refresh),
                window_days=args.calibration_window_days,
            )
        except Exception as e:
            print(f"Calibration artifact build failed: {e}")
            return
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

    if args.update_news:
        from news_loader import fetch_daily_news
        fetch_daily_news(force=True)
        return

    if args.tickers:
        initialize_db() # Ensure DB exists
        for ticker in args.tickers:
            analyze_ticker(ticker.upper())

if __name__ == "__main__":
    main()
