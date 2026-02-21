import glob
import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import requests

from config import (
    AUDIT_BENCHMARK_TICKER,
    ENABLE_DAILY_NOTIFICATIONS,
    ENABLE_WEEKLY_NOTIFICATIONS,
    NOTIFICATION_NTFY_TOPIC,
    NOTIFICATION_TIMEOUT_SECONDS,
    NOTIFICATION_WEBHOOK_URL,
    PUBLIC_REPORTS_DIR,
)


def _find_latest_summary_path() -> Optional[str]:
    pattern = os.path.join(PUBLIC_REPORTS_DIR, "model_performance_*.json")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _load_summary(path: Optional[str]) -> Optional[Dict]:
    summary_path = path or _find_latest_summary_path()
    if not summary_path or not os.path.exists(summary_path):
        return None
    try:
        with open(summary_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _build_message(summary: Dict) -> str:
    bench = summary.get("benchmark_ticker", AUDIT_BENCHMARK_TICKER)
    horizon_days = summary.get("horizon_days", 5)
    beat_rate = float(summary.get("beat_rate", 0.0))
    excess = float(summary.get("total_excess_return", 0.0))
    info_ratio = float(summary.get("information_ratio", 0.0))
    aligned_dates = int(summary.get("aligned_dates", 0))
    generated_at = summary.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M"))

    direction = "outperforming" if excess >= 0 else "underperforming"
    return (
        f"[Market Model Daily] {generated_at}\n"
        f"Horizon: {horizon_days}d | Benchmark: {bench}\n"
        f"Model is {direction} by {excess:.2%} total excess return.\n"
        f"Beat rate: {beat_rate:.1%} | Information ratio: {info_ratio:.2f}\n"
        f"Aligned signal dates: {aligned_dates}"
    )


def _build_weekly_message(summary: Dict) -> str:
    """Build weekly notification text using latest public summary payload."""
    bench = summary.get("benchmark_ticker", AUDIT_BENCHMARK_TICKER)
    generated_at = summary.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
    recent = summary.get("recent_relative", []) or []
    window = recent[-5:] if len(recent) >= 5 else recent

    if not window:
        return (
            f"[Market Model Weekly] {generated_at}\n"
            f"Benchmark: {bench}\n"
            "No recent aligned periods available yet."
        )

    def _prod_minus_one(key: str) -> float:
        val = 1.0
        for row in window:
            try:
                val *= 1.0 + float(row.get(key, 0.0))
            except Exception:
                pass
        return val - 1.0

    weekly_strategy = _prod_minus_one("strategy_return")
    weekly_benchmark = _prod_minus_one("benchmark_return")
    weekly_excess = _prod_minus_one("excess_return")
    beat_rate = 0.0
    if window:
        beats = 0
        for row in window:
            try:
                beats += 1 if float(row.get("excess_return", 0.0)) > 0 else 0
            except Exception:
                pass
        beat_rate = beats / max(1, len(window))

    direction = "outperforming" if weekly_excess >= 0 else "underperforming"
    return (
        f"[Market Model Weekly] {generated_at}\n"
        f"Window: last {len(window)} aligned periods | Benchmark: {bench}\n"
        f"Model is {direction} by {weekly_excess:.2%} over this window.\n"
        f"Model: {weekly_strategy:.2%} | {bench}: {weekly_benchmark:.2%}\n"
        f"Beat rate in window: {beat_rate:.1%}"
    )


def send_daily_summary_notification(summary_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Send daily model-performance notification.
    Delivery channels are configured in config.py / env vars.
    """
    if not ENABLE_DAILY_NOTIFICATIONS:
        return False, "Notifications disabled (ENABLE_DAILY_NOTIFICATIONS=False)"

    summary = _load_summary(summary_path)
    if not summary:
        return False, "No public model summary found to send"

    message = _build_message(summary)
    sent_channels = []
    errors = []

    if NOTIFICATION_WEBHOOK_URL:
        try:
            payload = {
                "text": message,
                "source": "market-analyzer",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            resp = requests.post(
                NOTIFICATION_WEBHOOK_URL,
                json=payload,
                timeout=NOTIFICATION_TIMEOUT_SECONDS,
            )
            if 200 <= resp.status_code < 300:
                sent_channels.append("webhook")
            else:
                errors.append(f"webhook status {resp.status_code}")
        except Exception as e:
            errors.append(f"webhook error: {e}")

    if NOTIFICATION_NTFY_TOPIC:
        try:
            url = f"https://ntfy.sh/{NOTIFICATION_NTFY_TOPIC}"
            resp = requests.post(
                url,
                data=message.encode("utf-8"),
                headers={"Title": "Market Model Daily", "Priority": "default"},
                timeout=NOTIFICATION_TIMEOUT_SECONDS,
            )
            if 200 <= resp.status_code < 300:
                sent_channels.append("ntfy")
            else:
                errors.append(f"ntfy status {resp.status_code}")
        except Exception as e:
            errors.append(f"ntfy error: {e}")

    if sent_channels:
        return True, f"Notification sent via {', '.join(sent_channels)}"
    if errors:
        return False, "; ".join(errors)
    return False, "No notification channels configured"


def send_weekly_summary_notification(summary_path: Optional[str] = None) -> Tuple[bool, str]:
    """Send weekly model-performance summary notification."""
    if not ENABLE_WEEKLY_NOTIFICATIONS:
        return False, "Weekly notifications disabled (ENABLE_WEEKLY_NOTIFICATIONS=False)"

    summary = _load_summary(summary_path)
    if not summary:
        return False, "No public model summary found to send"

    message = _build_weekly_message(summary)
    sent_channels = []
    errors = []

    if NOTIFICATION_WEBHOOK_URL:
        try:
            payload = {
                "text": message,
                "source": "market-analyzer",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            resp = requests.post(
                NOTIFICATION_WEBHOOK_URL,
                json=payload,
                timeout=NOTIFICATION_TIMEOUT_SECONDS,
            )
            if 200 <= resp.status_code < 300:
                sent_channels.append("webhook")
            else:
                errors.append(f"webhook status {resp.status_code}")
        except Exception as e:
            errors.append(f"webhook error: {e}")

    if NOTIFICATION_NTFY_TOPIC:
        try:
            url = f"https://ntfy.sh/{NOTIFICATION_NTFY_TOPIC}"
            resp = requests.post(
                url,
                data=message.encode("utf-8"),
                headers={"Title": "Market Model Weekly", "Priority": "default"},
                timeout=NOTIFICATION_TIMEOUT_SECONDS,
            )
            if 200 <= resp.status_code < 300:
                sent_channels.append("ntfy")
            else:
                errors.append(f"ntfy status {resp.status_code}")
        except Exception as e:
            errors.append(f"ntfy error: {e}")

    if sent_channels:
        return True, f"Weekly notification sent via {', '.join(sent_channels)}"
    if errors:
        return False, "; ".join(errors)
    return False, "No notification channels configured"
