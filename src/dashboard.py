"""
Market Analyzer Dashboard - Remote Control Interface

A lightweight Flask web UI for controlling the market scanner remotely.
Runs on localhost only (127.0.0.1:5050) for security.

Access via SSH tunnel:
    ssh -L 5050:127.0.0.1:5050 <user>@<remote-host>
    Then open: http://localhost:5050

Features:
- Run Scan Now: Immediate full market scan
- Schedule 9:30 AM: Auto-runs at next 9:30 AM (±5 min window, skips if missed)
- Train Models: Trigger multi-horizon model training
- Live Status: Shows current operation status
"""

import os
import sys
import json
import time
import threading
import subprocess
import re
import ast
import importlib
import mimetypes
import duckdb
from datetime import datetime, timedelta
from flask import Flask, render_template_string, jsonify, request, send_file

# Prevent __pycache__ generation (helps cloud sync churn).
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
SSH_TUNNEL_TARGET = os.environ.get("MARKET_DASHBOARD_SSH_TARGET", "<user>@<remote-host>")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    from config import (
        AUDIT_BENCHMARK_TICKER,
        DASHBOARD_AUDIT_DEFAULT_DAYS,
        DASHBOARD_AUDIT_DEFAULT_TIME,
        DASHBOARD_WEEKLY_SUMMARY_DEFAULT_DAY,
        DASHBOARD_WEEKLY_SUMMARY_DEFAULT_TIME,
        PUBLIC_REPORTS_DIR,
    )
except Exception:
    AUDIT_BENCHMARK_TICKER = "SPY"
    PUBLIC_REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "public")
    DASHBOARD_AUDIT_DEFAULT_TIME = "18:30"
    DASHBOARD_AUDIT_DEFAULT_DAYS = [0, 1, 2, 3, 4]
    DASHBOARD_WEEKLY_SUMMARY_DEFAULT_TIME = "18:45"
    DASHBOARD_WEEKLY_SUMMARY_DEFAULT_DAY = 4


def _sanitize_days(days, fallback=None):
    """Normalize weekday list into sorted unique ints [0..6]."""
    fallback = fallback if fallback is not None else [0, 1, 2, 3, 4]
    out = []
    for d in (days or []):
        try:
            di = int(d)
        except Exception:
            continue
        if 0 <= di <= 6 and di not in out:
            out.append(di)
    out.sort()
    return out or list(fallback)


def _parse_hour_minute(time_str: str, default: str = "18:30") -> tuple[int, int]:
    raw = (time_str or "").strip() or default
    try:
        hour, minute = map(int, raw.split(":"))
    except Exception:
        hour, minute = map(int, default.split(":"))
    hour = max(0, min(23, hour))
    minute = max(0, min(59, minute))
    return hour, minute


def _next_run_at(hour: int, minute: int) -> datetime:
    """Return next local datetime at the specified hour/minute."""
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def _next_run_for_days(hour: int, minute: int, allowed_days=None, now: datetime | None = None) -> datetime:
    """Return next run datetime for provided time on allowed weekdays."""
    now = now or datetime.now()
    allow = _sanitize_days(allowed_days, fallback=[0, 1, 2, 3, 4])
    for offset in range(0, 14):
        candidate = (now + timedelta(days=offset)).replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            continue
        if candidate.weekday() in allow:
            return candidate
    return now + timedelta(days=1)

# Global state
_AUDIT_DAYS_DEFAULT = _sanitize_days(DASHBOARD_AUDIT_DEFAULT_DAYS, fallback=[0, 1, 2, 3, 4])
_AUDIT_HOUR_DEFAULT, _AUDIT_MIN_DEFAULT = _parse_hour_minute(DASHBOARD_AUDIT_DEFAULT_TIME, "18:30")
_WEEKLY_HOUR_DEFAULT, _WEEKLY_MIN_DEFAULT = _parse_hour_minute(DASHBOARD_WEEKLY_SUMMARY_DEFAULT_TIME, "18:45")
_WEEKLY_DAY_DEFAULT = _sanitize_days([DASHBOARD_WEEKLY_SUMMARY_DEFAULT_DAY], fallback=[4])[0]

STATE = {
    # Scan schedule
    'scheduled_time': None,
    'schedule_window_minutes': 5,
    'scan_days': [0, 1, 2, 3, 4], # Default M-F
    'scan_interval_hours': 0,  # 0 = disabled (daily only), >0 = repeat every N hours
    'scan_running': False,
    'scan_last_run': None,
    'scan_last_result': None,
    'scan_stale_tickers': None,
    'scan_failed_fetch_tickers': None,
    'scan_insufficient_history_tickers': None,

    # Training schedule (N-day interval)
    'train_scheduled_time': None, 
    'train_interval_days': 10,
    'train_days': [0, 1, 2, 3, 4, 5, 6], # Default All
    'train_running': False,
    'train_last_run': None,
    'train_last_result': None,
    
    # Qual features schedule
    'qual_chunk_size': 20,
    'qual_interval_hours': 4,
    'qual_scheduled_time': None,
    'qual_running': False,
    'qual_last_run': None,
    'qual_last_result': None,

    # Audit schedule (default market days)
    'audit_days': _AUDIT_DAYS_DEFAULT,
    'audit_time': f"{_AUDIT_HOUR_DEFAULT:02d}:{_AUDIT_MIN_DEFAULT:02d}",
    'audit_scheduled_time': _next_run_for_days(_AUDIT_HOUR_DEFAULT, _AUDIT_MIN_DEFAULT, _AUDIT_DAYS_DEFAULT),
    'audit_running': False,
    'audit_last_run': None,
    'audit_last_result': None,

    # Weekly summary schedule
    'weekly_summary_day': _WEEKLY_DAY_DEFAULT,
    'weekly_summary_time': f"{_WEEKLY_HOUR_DEFAULT:02d}:{_WEEKLY_MIN_DEFAULT:02d}",
    'weekly_summary_scheduled_time': _next_run_for_days(_WEEKLY_HOUR_DEFAULT, _WEEKLY_MIN_DEFAULT, [_WEEKLY_DAY_DEFAULT]),
    'weekly_summary_running': False,
    'weekly_summary_last_result': None,

    # News assessment schedule (morning + evening Gemini CLI calls)
    'news_morning_time': '08:30',
    'news_evening_time': '16:30',
    'news_days': [0, 1, 2, 3, 4],  # Default M-F
    'news_scheduled_time': None,
    'news_running': False,
    'news_last_run': None,
    'news_last_result': None,

    # Analytics snapshot scheduler
    'snapshot_interval_minutes': 45,
    'snapshot_scheduled_time': None,
    'snapshot_recent_days': 30,
    'snapshot_running': False,
    'snapshot_last_run': None,
    'snapshot_last_result': None,

    # Legacy compat
    'last_result': None,
}
STATE_LOCK = threading.Lock()

# Process management - separate processes for each operation
PROCESSES = {
    'scan': None,
    'train': None,
    'qual': None,
    'news': None,
    'audit': None,
    'weekly_summary': None,
    'snapshot': None,
}
PROCESS_LOCK = threading.Lock()
SCAN_SCHEDULER_STARTED = False
NEWS_SCHEDULER_STARTED = False
AUDIT_SCHEDULER_STARTED = False
WEEKLY_SUMMARY_SCHEDULER_STARTED = False
SNAPSHOT_SCHEDULER_STARTED = False
PUBLIC_DASHBOARD_PATH = os.path.join(
    PUBLIC_REPORTS_DIR, f"model_performance_{AUDIT_BENCHMARK_TICKER.lower()}.html"
)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.py")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "market_data.duckdb")
CONFIG_HISTORY_PATH = os.path.join(PROJECT_ROOT, "data", "config_history.jsonl")
SCAN_SCHEDULE_STATE_PATH = os.path.join(PROJECT_ROOT, "data", "scan_schedule_state.json")
try:
    SCAN_LOCK_RETRY_MINUTES = max(1, int(os.environ.get("DASHBOARD_SCAN_LOCK_RETRY_MINUTES", "15")))
except ValueError:
    SCAN_LOCK_RETRY_MINUTES = 15

EDITABLE_CONFIG_KEYS = (
    "PORTFOLIO_HOLDINGS",
    "WATCHLIST",
    "UNIVERSE_DENYLIST",
    "CONFLUENCE_THRESHOLD",
    "TACTICAL_THRESHOLD",
    "TREND_THRESHOLD",
    "TRADABILITY_MIN_PRICE",
    "TRADABILITY_MIN_AVG_VOLUME_20D",
    "TRADABILITY_MIN_AVG_DOLLAR_VOLUME_20D",
    "TRADABILITY_MAX_ATR_RATIO",
    "HORIZON_TARGETS",
)


def _normalize_ticker_list(raw) -> list[str]:
    """Normalize ticker list input from textarea/json into uppercase symbols."""
    if raw is None:
        return []
    if isinstance(raw, str):
        tokens = re.split(r"[\s,;]+", raw.strip())
    elif isinstance(raw, list):
        tokens = [str(x).strip() for x in raw]
    else:
        tokens = [str(raw).strip()]
    out = []
    seen = set()
    for token in tokens:
        ticker = token.strip().upper()
        if not ticker:
            continue
        if ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return out


def _safe_float(raw, default=None):
    try:
        return float(raw)
    except Exception:
        return default


def _safe_int(raw, default=None):
    try:
        return int(float(raw))
    except Exception:
        return default


def _load_config_module():
    """Load and reload config module so dashboard reflects latest saved values."""
    import config as cfg  # local import to avoid startup failure before sys.path setup

    return importlib.reload(cfg)


def _config_snapshot() -> dict:
    """Return dashboard-editable config snapshot from runtime module."""
    cfg = _load_config_module()
    horizon_targets = dict(getattr(cfg, "HORIZON_TARGETS", {}) or {})
    return {
        "portfolio_holdings": list(getattr(cfg, "PORTFOLIO_HOLDINGS", []) or []),
        "watchlist": list(getattr(cfg, "WATCHLIST", []) or []),
        "universe_denylist": list(getattr(cfg, "UNIVERSE_DENYLIST", []) or []),
        "confluence_threshold": float(getattr(cfg, "CONFLUENCE_THRESHOLD", 0.60)),
        "tactical_threshold": float(getattr(cfg, "TACTICAL_THRESHOLD", 0.70)),
        "trend_threshold": float(getattr(cfg, "TREND_THRESHOLD", 0.65)),
        "tradability_min_price": float(getattr(cfg, "TRADABILITY_MIN_PRICE", 3.0)),
        "tradability_min_avg_volume_20d": int(getattr(cfg, "TRADABILITY_MIN_AVG_VOLUME_20D", 300000)),
        "tradability_min_avg_dollar_volume_20d": int(getattr(cfg, "TRADABILITY_MIN_AVG_DOLLAR_VOLUME_20D", 5000000)),
        "tradability_max_atr_ratio": float(getattr(cfg, "TRADABILITY_MAX_ATR_RATIO", 0.20)),
        "horizon_targets": {
            "5": float(horizon_targets.get(5, horizon_targets.get("5", 0.015))),
            "10": float(horizon_targets.get(10, horizon_targets.get("10", 0.030))),
            "30": float(horizon_targets.get(30, horizon_targets.get("30", 0.080))),
        },
    }


def _render_assignment(name: str, value) -> str:
    """Render assignment block for direct replacement in config.py."""
    if isinstance(value, list):
        lines = [f"{name} = ["]
        for item in value:
            lines.append(f"    {item!r},")
        lines.append("]")
        return "\n".join(lines) + "\n"
    if isinstance(value, dict):
        keys = sorted(value.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        lines = [f"{name} = {{"]
        for key in keys:
            k = int(key) if str(key).isdigit() else key
            lines.append(f"    {k!r}: {float(value[key]):.6f},")
        lines.append("}")
        return "\n".join(lines) + "\n"
    if isinstance(value, float):
        return f"{name} = {value}\n"
    if isinstance(value, int):
        return f"{name} = {value}\n"
    return f"{name} = {value!r}\n"


def _replace_assignment_block(file_text: str, name: str, assignment_block: str) -> str:
    """Replace a top-level assignment in config.py using AST line boundaries."""
    tree = ast.parse(file_text)
    lines = file_text.splitlines(keepends=True)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t for t in node.targets if isinstance(t, ast.Name)]
        if not targets:
            continue
        if not any(t.id == name for t in targets):
            continue
        start = node.lineno - 1
        end = node.end_lineno or node.lineno
        lines[start:end] = [assignment_block]
        return "".join(lines)
    raise KeyError(f"Assignment not found in config.py: {name}")


def _update_config_file(updates: dict) -> list[str]:
    """Apply selected config updates to config.py atomically."""
    if not updates:
        return []
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    changed_keys = []
    for key, value in updates.items():
        if key not in EDITABLE_CONFIG_KEYS:
            continue
        block = _render_assignment(key, value)
        text = _replace_assignment_block(text, key, block)
        changed_keys.append(key)
    tmp_path = f"{CONFIG_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, CONFIG_PATH)
    return changed_keys


def _list_files(base_dir: str, exts: tuple[str, ...], max_files: int = 120) -> list[dict]:
    """Return recent files for reports/log browser."""
    items = []
    if not os.path.isdir(base_dir):
        return items
    for root, _, files in os.walk(base_dir):
        for name in files:
            if exts and not name.lower().endswith(exts):
                continue
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, base_dir)
            try:
                st = os.stat(abs_path)
                items.append(
                    {
                        "name": name,
                        "rel_path": rel_path.replace("\\", "/"),
                        "size": int(st.st_size),
                        "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
            except Exception:
                continue
    items.sort(key=lambda x: x.get("mtime", ""), reverse=True)
    return items[:max_files]


def _safe_join(base_dir: str, rel_path: str) -> str:
    """Resolve and validate file path to prevent path traversal."""
    safe_rel = str(rel_path or "").strip().lstrip("/").replace("\\", "/")
    if not safe_rel:
        raise ValueError("Missing file path")
    candidate = os.path.abspath(os.path.join(base_dir, safe_rel))
    base_abs = os.path.abspath(base_dir)
    if not (candidate == base_abs or candidate.startswith(base_abs + os.sep)):
        raise ValueError("Invalid file path")
    if not os.path.isfile(candidate):
        raise FileNotFoundError(candidate)
    return candidate


def _resolve_kind_base_dir(kind: str) -> str:
    kind = (kind or "").strip().lower()
    if kind == "reports":
        return REPORTS_DIR
    if kind == "logs":
        return LOGS_DIR
    if kind == "public":
        return PUBLIC_REPORTS_DIR
    raise ValueError("Invalid file kind")


def _file_preview(kind: str, rel_path: str, max_chars: int = 30000) -> dict:
    """Read safe text preview for a report/log file."""
    base_dir = _resolve_kind_base_dir(kind)
    abs_path = _safe_join(base_dir, rel_path)
    ext = os.path.splitext(abs_path)[1].lower()
    mime, _ = mimetypes.guess_type(abs_path)
    is_text_like = (
        ext in {".txt", ".log", ".md", ".json", ".csv", ".html", ".py", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
        or (mime or "").startswith("text/")
        or mime in {"application/json", "application/xml"}
    )
    if not is_text_like:
        raise ValueError("Preview supports text-like files only")

    max_chars = max(1000, min(120000, int(max_chars or 30000)))
    with open(abs_path, "rb") as f:
        raw = f.read(max_chars + 1)
    truncated = len(raw) > max_chars
    if truncated:
        raw = raw[:max_chars]
    content = raw.decode("utf-8", errors="replace")
    st = os.stat(abs_path)
    return {
        "kind": kind,
        "rel_path": os.path.relpath(abs_path, base_dir).replace("\\", "/"),
        "size": int(st.st_size),
        "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "mime": mime or "text/plain",
        "truncated": truncated,
        "content": content,
    }


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _append_config_history(entry: dict) -> None:
    """Append one config history entry as JSONL."""
    _ensure_parent_dir(CONFIG_HISTORY_PATH)
    with open(CONFIG_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, separators=(",", ":")))
        f.write("\n")


def _read_config_history(limit: int = 50) -> list[dict]:
    """Read config history newest-first."""
    if not os.path.exists(CONFIG_HISTORY_PATH):
        return []
    rows = []
    with open(CONFIG_HISTORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    rows.reverse()
    return rows[: max(1, int(limit))]


def _dashboard_snapshot_to_config_updates(snapshot: dict) -> dict:
    """Convert dashboard snapshot payload into config assignment updates."""
    if not isinstance(snapshot, dict):
        return {}
    updates = {}
    if "portfolio_holdings" in snapshot:
        updates["PORTFOLIO_HOLDINGS"] = _normalize_ticker_list(snapshot.get("portfolio_holdings"))
    if "watchlist" in snapshot:
        updates["WATCHLIST"] = _normalize_ticker_list(snapshot.get("watchlist"))
    if "universe_denylist" in snapshot:
        updates["UNIVERSE_DENYLIST"] = _normalize_ticker_list(snapshot.get("universe_denylist"))

    float_map = {
        "confluence_threshold": "CONFLUENCE_THRESHOLD",
        "tactical_threshold": "TACTICAL_THRESHOLD",
        "trend_threshold": "TREND_THRESHOLD",
        "tradability_min_price": "TRADABILITY_MIN_PRICE",
        "tradability_max_atr_ratio": "TRADABILITY_MAX_ATR_RATIO",
    }
    int_map = {
        "tradability_min_avg_volume_20d": "TRADABILITY_MIN_AVG_VOLUME_20D",
        "tradability_min_avg_dollar_volume_20d": "TRADABILITY_MIN_AVG_DOLLAR_VOLUME_20D",
    }
    for src_key, dst_key in float_map.items():
        if src_key in snapshot:
            val = _safe_float(snapshot.get(src_key), None)
            if val is not None:
                updates[dst_key] = val
    for src_key, dst_key in int_map.items():
        if src_key in snapshot:
            val = _safe_int(snapshot.get(src_key), None)
            if val is not None:
                updates[dst_key] = val

    if "horizon_targets" in snapshot and isinstance(snapshot.get("horizon_targets"), dict):
        incoming = snapshot.get("horizon_targets") or {}
        updates["HORIZON_TARGETS"] = {
            "5": _safe_float(incoming.get("5"), 0.015),
            "10": _safe_float(incoming.get("10"), 0.030),
            "30": _safe_float(incoming.get("30"), 0.080),
        }
    return updates


def _latest_log_path(prefix: str) -> str:
    """Return latest log file path by mtime for a given filename prefix."""
    if not os.path.isdir(LOGS_DIR):
        return ""
    candidates = []
    for name in os.listdir(LOGS_DIR):
        if not name.startswith(prefix) or not name.endswith(".log"):
            continue
        path = os.path.join(LOGS_DIR, name)
        try:
            st = os.stat(path)
            candidates.append((st.st_mtime, path))
        except Exception:
            continue
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _tail_text(path: str, max_lines: int = 160) -> str:
    """Return tail text from file safely."""
    if not path or not os.path.exists(path):
        return ""
    try:
        proc = subprocess.run(
            ["tail", "-n", str(max(1, int(max_lines))), path],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        return proc.stdout or ""
    except subprocess.TimeoutExpired:
        return ""
    except Exception:
        return ""


def _parse_log_phase(full_tail: str) -> str:
    """Extract latest phase marker from full-run log tail."""
    if not full_tail:
        return ""
    phase = ""
    for line in full_tail.splitlines():
        m = re.search(r"---\s*Phase\s+\d+/\d+:\s*([A-Za-z0-9_\- ]+)\s*---", line)
        if m:
            phase = m.group(1).strip()
    return phase


def _extract_marker_int(text: str, key: str) -> int | None:
    """Extract int marker from text like 'integration_rc=0'."""
    if not text:
        return None
    m = re.search(rf"{re.escape(key)}\s*=\s*(-?\d+)", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _normalize_rows_per_min(raw: dict | None) -> dict:
    """Normalize throughput windows and provide a usable 10m fallback."""
    out = {"5m": 0.0, "10m": 0.0, "15m": 0.0, "20m": 0.0, "60m": 0.0}
    if isinstance(raw, dict):
        for key, value in raw.items():
            key_str = str(key)
            if key_str not in out:
                continue
            try:
                out[key_str] = float(value)
            except Exception:
                continue
    if out["10m"] <= 0:
        if out["15m"] > 0:
            out["10m"] = out["15m"]
        elif out["5m"] > 0:
            out["10m"] = out["5m"]
        elif out["20m"] > 0:
            out["10m"] = out["20m"]
        elif out["60m"] > 0:
            out["10m"] = out["60m"]
    return out


def _recent_snapshot_history(limit: int = 12) -> list[dict]:
    """Return compact history from analytics snapshot JSON files."""
    if not os.path.isdir(REPORTS_DIR):
        return []
    candidates = []
    for name in os.listdir(REPORTS_DIR):
        if name.startswith("analytics_snapshot_") and name.endswith(".json"):
            path = os.path.join(REPORTS_DIR, name)
            try:
                candidates.append((os.path.getmtime(path), path))
            except Exception:
                continue
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[0], reverse=True)
    items = []
    for _, path in candidates[: max(1, int(limit))]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                snap = json.load(f)
            preds = snap.get("model_predictions") or {}
            throughput = _normalize_rows_per_min(preds.get("throughput_rows_per_min") or {})
            items.append(
                {
                    "path": path,
                    "generated_at_utc": str(snap.get("generated_at_utc") or ""),
                    "rows_total": int(preds.get("rows_total") or 0),
                    "rows_per_min": throughput,
                    "latest_model_hash": str(preds.get("latest_model_hash") or ""),
                }
            )
        except Exception:
            continue
    return items


def _parse_process_line(line: str) -> dict:
    """Parse `ps -axo pid,pcpu,pmem,state,etime,command` row."""
    parts = (line or "").strip().split(None, 5)
    if len(parts) < 6:
        return {}
    pid_s, cpu_s, mem_s, state, elapsed, command = parts
    try:
        pid = int(pid_s)
    except Exception:
        pid = None
    try:
        cpu = float(cpu_s)
    except Exception:
        cpu = None
    try:
        mem = float(mem_s)
    except Exception:
        mem = None
    return {
        "pid": pid,
        "cpu_percent": cpu,
        "mem_percent": mem,
        "state": state,
        "elapsed": elapsed,
        "command": command,
    }


def _collect_db_run_stats(timeout_seconds: float = 2.5) -> dict:
    """
    Collect DB run stats from latest analytics snapshot JSON (read-only, non-blocking).
    """
    _ = timeout_seconds  # compatibility placeholder
    default = {
        "db_exists": os.path.exists(DB_PATH),
        "price_rows": 0,
        "price_tickers": 0,
        "price_min_date": "",
        "price_max_date": "",
        "pred_rows_total": 0,
        "pred_latest_hash": "",
        "pred_max_created_at": "",
        "rows_per_min": {"5m": 0.0, "10m": 0.0, "15m": 0.0, "20m": 0.0, "60m": 0.0},
        "latest_hash_horizon_stats": [],
        "replay_progress": {},
    }
    candidates = []
    if os.path.isdir(REPORTS_DIR):
        for name in os.listdir(REPORTS_DIR):
            if name.startswith("analytics_snapshot_") and name.endswith(".json"):
                path = os.path.join(REPORTS_DIR, name)
                try:
                    candidates.append((os.path.getmtime(path), path))
                except Exception:
                    continue
    if not candidates:
        default["error"] = "no analytics snapshot found; run scripts/generate_analytics_snapshot.py"
        return default
    candidates.sort(key=lambda x: x[0], reverse=True)
    snapshot_path = candidates[0][1]
    try:
        with open(snapshot_path, "r", encoding="utf-8") as f:
            snap = json.load(f)
        price = snap.get("price_history") or {}
        preds = snap.get("model_predictions") or {}
        default.update(
            {
                "price_rows": int(price.get("rows") or 0),
                "price_tickers": int(price.get("tickers") or 0),
                "price_min_date": str(price.get("min_date") or ""),
                "price_max_date": str(price.get("max_date") or ""),
                "pred_rows_total": int(preds.get("rows_total") or 0),
                "pred_latest_hash": str(preds.get("latest_model_hash") or ""),
                "pred_max_created_at": str(preds.get("max_created_at") or ""),
                "rows_per_min": _normalize_rows_per_min(preds.get("throughput_rows_per_min") or {}),
                "latest_hash_horizon_stats": list(preds.get("latest_hash_by_horizon") or []),
                "replay_progress": {
                    "rows_done": int(preds.get("rows_total") or 0),
                    "dates_done": None,
                    "max_asof_date": (
                        (preds.get("latest_hash_by_horizon") or [{}])[0].get("max_asof_date", "")
                        if preds.get("latest_hash_by_horizon")
                        else ""
                    ),
                    "progress_pct": None,
                    "eta_minutes": None,
                },
                "snapshot_path": snapshot_path,
                "snapshot_generated_at_utc": str(snap.get("generated_at_utc") or ""),
            }
        )
        return default
    except Exception as exc:
        default["error"] = f"snapshot parse failed: {exc}"
        return default


def _collect_run_monitor() -> dict:
    """Build run monitor payload from logs/process/db."""
    full_log = _latest_log_path("full_run_")
    post_log = _latest_log_path("postcheck_")
    # Avoid expensive tail reads on cloud-synced active logs; rely on process/db state.
    full_tail = ""
    post_tail = ""

    process_line = ""
    process_info = {}
    process_running = False
    try:
        ps = subprocess.run(
            ["ps", "-axo", "pid,pcpu,pmem,state,etime,command"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in (ps.stdout or "").splitlines():
            if "src/main.py --all" in line or "src/main.py --pipeline full" in line:
                process_line = line.strip()
                process_info = _parse_process_line(process_line)
                process_running = True
                break
    except Exception:
        pass

    full_done = False
    post_done = False
    full_mtime = ""
    post_mtime = ""
    try:
        if full_log and os.path.exists(full_log):
            full_mtime = datetime.fromtimestamp(os.path.getmtime(full_log)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        full_mtime = ""
    try:
        if post_log and os.path.exists(post_log):
            post_mtime = datetime.fromtimestamp(os.path.getmtime(post_log)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        post_mtime = ""

    phase = "RUNNING" if process_running else "IDLE"
    if process_info.get("state"):
        phase = f"{phase} ({process_info.get('state')})"

    snapshot_history = _recent_snapshot_history(limit=12)
    return {
        "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "process_running": process_running,
        "process_line": process_line,
        "process": process_info,
        "phase": phase,
        "full_log": {
            "path": full_log,
            "exists": bool(full_log and os.path.exists(full_log)),
            "mtime": full_mtime,
            "done": full_done,
            "tail": full_tail[-4000:],
            "end_rc": _extract_marker_int(full_tail, "END full pipeline rc"),
        },
        "post_log": {
            "path": post_log,
            "exists": bool(post_log and os.path.exists(post_log)),
            "mtime": post_mtime,
            "done": post_done,
            "tail": post_tail[-4000:],
            "full_run_rc": _extract_marker_int(post_tail, "full_run_rc"),
            "integration_rc": _extract_marker_int(post_tail, "integration_rc"),
            "runtime_skew_rc": _extract_marker_int(post_tail, "runtime_skew_rc"),
            "end_rc": _extract_marker_int(post_tail, "END post-check rc"),
        },
        "db": _collect_db_run_stats(),
        "snapshot_history": snapshot_history,
    }

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Market Analyzer Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        :root {
            --bg-0: #f3f7fb;
            --bg-1: #e8eff8;
            --ink-strong: #102337;
            --ink: #23384d;
            --muted: #607086;
            --card: rgba(255, 255, 255, 0.92);
            --card-border: #d8e3ef;
            --accent: #0f8b8d;
            --accent-2: #0ea5e9;
            --warn: #f59e0b;
            --danger: #ef4444;
            --ok: #22c55e;
            --shadow: 0 10px 30px rgba(11, 32, 53, 0.08);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: "Avenir Next", "IBM Plex Sans", "Source Sans 3", "Segoe UI", sans-serif;
            background:
                radial-gradient(1200px 700px at 5% 5%, #ffffff 0%, transparent 60%),
                radial-gradient(1200px 700px at 95% -5%, #dff4f2 0%, transparent 55%),
                linear-gradient(155deg, var(--bg-0) 0%, var(--bg-1) 100%);
            min-height: 100vh;
            color: var(--ink);
            padding: 24px;
        }
        .container { max-width: 1440px; margin: 0 auto; }
        .topbar {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 6px;
            margin-bottom: 22px;
            padding: 10px 4px;
        }
        h1 {
            margin: 0;
            font-size: clamp(2rem, 2.7vw, 2.8rem);
            letter-spacing: -0.02em;
            color: var(--ink-strong);
            line-height: 1.05;
        }
        .subtle {
            color: var(--muted);
            font-size: 0.82rem;
        }
        .summary-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }
        .summary-tile {
            background: linear-gradient(160deg, rgba(255, 255, 255, 0.95) 0%, rgba(241, 248, 255, 0.95) 100%);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 12px 14px;
            box-shadow: var(--shadow);
            min-height: 88px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 5px;
        }
        .summary-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            font-weight: 700;
        }
        .summary-value {
            font-size: 1.22rem;
            color: var(--ink-strong);
            font-weight: 700;
            line-height: 1.1;
        }
        .summary-note {
            font-size: 0.78rem;
            color: var(--muted);
            line-height: 1.2;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(12, minmax(0, 1fr));
            gap: 16px;
            align-items: start;
        }
        .card {
            background: var(--card);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid var(--card-border);
            box-shadow: var(--shadow);
        }
        .card-full { grid-column: 1 / -1; }
        .card-wide { grid-column: span 8; }
        .card-compact { grid-column: span 4; }
        .card h2 {
            font-size: 0.9rem;
            margin-bottom: 14px;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 10px 14px;
            font-size: 0.93rem;
            font-weight: 700;
            border: 1px solid transparent;
            border-radius: 10px;
            cursor: pointer;
            transition: transform 0.15s ease, box-shadow 0.2s ease, background 0.2s ease;
            margin: 0;
            min-width: 132px;
            color: #fff;
            text-decoration: none;
        }
        .btn:hover { transform: translateY(-1px); }
        .btn-primary {
            background: linear-gradient(135deg, var(--accent) 0%, #0b7285 100%);
            box-shadow: 0 6px 14px rgba(15, 139, 141, 0.25);
        }
        .btn-secondary {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            box-shadow: 0 6px 14px rgba(37, 99, 235, 0.2);
        }
        .btn-warning {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            box-shadow: 0 6px 14px rgba(217, 119, 6, 0.2);
        }
        .btn-danger {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            box-shadow: 0 6px 14px rgba(220, 38, 38, 0.2);
        }
        .btn:disabled {
            opacity: 0.55;
            cursor: not-allowed;
            transform: none !important;
            box-shadow: none !important;
        }
        .status {
            padding: 14px;
            border-radius: 10px;
            margin-top: 12px;
            font-family: "IBM Plex Mono", Menlo, Consolas, monospace;
            font-size: 0.82rem;
            font-weight: 500;
        }
        .status-idle { background: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.35); color: #166534; }
        .status-running { background: rgba(245, 158, 11, 0.16); border: 1px solid rgba(245, 158, 11, 0.4); color: #854d0e; }
        .status-scheduled { background: rgba(14, 165, 233, 0.12); border: 1px solid rgba(14, 165, 233, 0.35); color: #0c4a6e; }
        .schedule-form { display: flex; flex-direction: column; gap: 12px; }
        .schedule-row {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .schedule-form input[type=time], .schedule-form input[type=number] {
            padding: 9px 10px;
            border-radius: 8px;
            border: 1px solid #ced9e5;
            background: #fff;
            color: var(--ink-strong);
            font-size: 0.92rem;
        }
        .schedule-form input:focus,
        .field input:focus,
        .field textarea:focus,
        .field select:focus {
            outline: none;
            border-color: #7cc4f2;
            box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.16);
        }
        .field-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .field {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .field label {
            font-size: 0.74rem;
            color: var(--muted);
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
        }
        .field input, .field textarea, .field select {
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ced9e5;
            background: #fff;
            color: var(--ink-strong);
            font-size: 0.92rem;
        }
        .field textarea {
            min-height: 88px;
            resize: vertical;
            font-family: "IBM Plex Mono", Menlo, Consolas, monospace;
        }
        .file-list {
            margin-top: 8px;
            max-height: 250px;
            overflow: auto;
            border: 1px solid #d8e3ef;
            border-radius: 10px;
            background: #f8fbff;
            padding: 6px;
        }
        .file-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 8px;
            align-items: center;
            padding: 8px;
            border-bottom: 1px dashed #d5dfeb;
        }
        .file-row:last-child { border-bottom: none; }
        .file-meta {
            font-size: 0.76rem;
            color: var(--muted);
            margin-top: 2px;
        }
        .file-actions a {
            color: #0369a1;
            text-decoration: none;
            font-size: 0.82rem;
            font-weight: 700;
        }
        .file-actions button {
            margin-left: 8px;
            padding: 4px 8px;
            border-radius: 7px;
            border: 1px solid #c9d8ea;
            background: #fff;
            color: #0c4a6e;
            font-size: 0.76rem;
            font-weight: 700;
            cursor: pointer;
        }
        .file-actions button:hover {
            background: #eef6ff;
        }
        .file-actions a:hover { text-decoration: underline; }
        .preview-shell {
            margin-top: 8px;
            border: 1px solid #d8e3ef;
            background: #f8fbff;
            border-radius: 10px;
            overflow: hidden;
        }
        .preview-header {
            padding: 8px 10px;
            border-bottom: 1px solid #d8e3ef;
            font-size: 0.76rem;
            color: var(--muted);
            font-weight: 700;
            background: #eef5fd;
        }
        .preview-content {
            margin: 0;
            padding: 10px;
            max-height: 420px;
            overflow: auto;
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 0.78rem;
            line-height: 1.45;
            color: #0f2034;
            font-family: "IBM Plex Mono", Menlo, Consolas, monospace;
        }
        .days-selector {
            display: flex;
            gap: 8px;
            background: #f3f8ff;
            padding: 8px;
            border-radius: 10px;
            border: 1px solid #d9e5f2;
        }
        .day-check {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 32px;
            padding: 4px 2px;
            border-radius: 8px;
            background: #fff;
            border: 1px solid #d7e2ee;
            cursor: pointer;
            font-size: 0.75rem;
            color: var(--muted);
        }
        .day-check input {
            margin-bottom: 2px;
            cursor: pointer;
            accent-color: var(--accent-2);
        }
        .day-check span { font-weight: 700; }
        .time-display {
            font-size: clamp(1.5rem, 2vw, 2.1rem);
            font-weight: 650;
            color: var(--ink-strong);
            text-align: left;
            margin-bottom: 6px;
            letter-spacing: -0.01em;
        }
        .info {
            font-size: 0.8rem;
            color: var(--muted);
            margin-top: 8px;
            line-height: 1.35;
        }
        .monitor-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin-top: 8px;
        }
        .monitor-metric {
            background: #f8fbff;
            border: 1px solid #d8e3ef;
            border-radius: 10px;
            padding: 10px;
        }
        .monitor-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: var(--muted);
            margin-bottom: 4px;
            font-weight: 700;
        }
        .monitor-value {
            font-size: 1rem;
            color: var(--ink-strong);
            font-weight: 700;
        }
        .monitor-controls {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 10px;
            margin-top: 10px;
            align-items: end;
        }
        .monitor-controls .field {
            gap: 4px;
        }
        .trend-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }
        .trend-card {
            background: #f8fbff;
            border: 1px solid #d8e3ef;
            border-radius: 10px;
            padding: 10px;
        }
        .trend-title {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: var(--muted);
            margin-bottom: 6px;
            font-weight: 700;
        }
        .trend-svg {
            display: block;
            width: 100%;
            height: 90px;
            border-radius: 8px;
            background: linear-gradient(180deg, rgba(14, 165, 233, 0.10) 0%, rgba(14, 165, 233, 0.02) 100%);
        }
        .trend-caption {
            margin-top: 6px;
            font-size: 0.76rem;
            color: var(--muted);
        }
        .history-list {
            margin-top: 8px;
            max-height: 180px;
            overflow: auto;
            border: 1px solid #d8e3ef;
            border-radius: 10px;
            background: #f8fbff;
            padding: 6px;
        }
        .history-item {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 8px;
            align-items: center;
            padding: 8px;
            border-bottom: 1px dashed #d5dfeb;
        }
        .history-item:last-child { border-bottom: none; }
        .history-meta {
            font-size: 0.76rem;
            color: var(--muted);
            margin-top: 2px;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.65; } }
        .running { animation: pulse 1.4s infinite; }
        @media (max-width: 1200px) {
            .summary-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .card-wide { grid-column: span 7; }
            .card-compact { grid-column: span 5; }
        }
        @media (max-width: 960px) {
            body { padding: 14px; }
            .summary-strip { grid-template-columns: 1fr; }
            .dashboard-grid { grid-template-columns: 1fr; }
            .card-full, .card-wide, .card-compact { grid-column: 1 / -1; }
            .field-grid { grid-template-columns: 1fr; }
            .topbar { align-items: flex-start; }
            .monitor-grid { grid-template-columns: 1fr 1fr; }
            .monitor-controls { grid-template-columns: 1fr 1fr; }
            .trend-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="topbar">
            <h1>📊 Market Analyzer</h1>
            <p class="subtle">Dashboard PID: {{ pid }}</p>
        </div>
        <div class="summary-strip">
            <div class="summary-tile">
                <div class="summary-label">Active Ops</div>
                <div class="summary-value" id="sumOps">0</div>
                <div class="summary-note" id="sumOpsNote">Idle</div>
            </div>
            <div class="summary-tile">
                <div class="summary-label">Next Scan</div>
                <div class="summary-value" id="sumNextScan">-</div>
                <div class="summary-note" id="sumNextScanNote">Not scheduled</div>
            </div>
            <div class="summary-tile">
                <div class="summary-label">Pipeline Process</div>
                <div class="summary-value" id="sumProc">-</div>
                <div class="summary-note" id="sumProcNote">No active process</div>
            </div>
            <div class="summary-tile">
                <div class="summary-label">Latest Snapshot</div>
                <div class="summary-value" id="sumSnapshot">-</div>
                <div class="summary-note" id="sumSnapshotNote">No snapshot yet</div>
            </div>
        </div>
        <div class="dashboard-grid">
        
        <div class="card card-full">
            <div class="time-display" id="clock"></div>
            <div id="runningOps" style="margin-top:10px;"></div>
        </div>

        <div class="card card-full">
            <h2>Live Run Monitor</h2>
            <div class="schedule-row" style="margin-bottom:8px;">
                <button class="btn btn-secondary" onclick="loadRunMonitor()">↻ Refresh Monitor</button>
                <a class="btn btn-secondary" id="fullLogLink" href="#" target="_blank" rel="noopener">Open Full Log</a>
                <a class="btn btn-secondary" id="postLogLink" href="#" target="_blank" rel="noopener">Open Post-Check</a>
            </div>
            <div class="monitor-controls">
                <div class="field">
                    <label for="snapshotIntervalMinutes">Snapshot Interval (min)</label>
                    <input type="number" id="snapshotIntervalMinutes" min="5" max="720" value="45">
                </div>
                <div class="field">
                    <label for="snapshotRecentDays">Recent Days</label>
                    <input type="number" id="snapshotRecentDays" min="1" max="365" value="30">
                </div>
                <div class="field">
                    <button class="btn btn-primary" onclick="runSnapshotNow()" id="snapshotBtn">📸 Snapshot Now</button>
                </div>
                <div class="field">
                    <button class="btn btn-danger" onclick="stopOp('snapshot')" id="stopSnapshotBtn" style="display:none;">🛑 Stop Snapshot</button>
                </div>
                <div class="field">
                    <button class="btn btn-warning" onclick="setSnapshotSchedule()" id="snapshotScheduleBtn">⏱ Auto Snapshot</button>
                </div>
                <div class="field">
                    <button class="btn btn-danger" onclick="cancelSnapshotSchedule()" id="snapshotCancelBtn" style="display:none;">✖ Cancel Auto</button>
                </div>
            </div>
            <div id="snapshotScheduleStatus" class="info"></div>
            <div id="runMonitorStatus" class="info"></div>
            <div class="monitor-grid">
                <div class="monitor-metric"><div class="monitor-label">Phase</div><div class="monitor-value" id="monPhase">-</div></div>
                <div class="monitor-metric"><div class="monitor-label">Process</div><div class="monitor-value" id="monProc">-</div></div>
                <div class="monitor-metric"><div class="monitor-label">Rows/Min (10m)</div><div class="monitor-value" id="monRpm10">-</div></div>
                <div class="monitor-metric"><div class="monitor-label">ETA (Replay)</div><div class="monitor-value" id="monEta">-</div></div>
                <div class="monitor-metric"><div class="monitor-label">Progress</div><div class="monitor-value" id="monProgress">-</div></div>
                <div class="monitor-metric"><div class="monitor-label">Rows Done</div><div class="monitor-value" id="monRowsDone">-</div></div>
                <div class="monitor-metric"><div class="monitor-label">Latest As-Of</div><div class="monitor-value" id="monAsOf">-</div></div>
                <div class="monitor-metric"><div class="monitor-label">Latest Hash</div><div class="monitor-value" id="monHash">-</div></div>
                <div class="monitor-metric"><div class="monitor-label">CPU %</div><div class="monitor-value" id="monCpu">-</div></div>
                <div class="monitor-metric"><div class="monitor-label">Memory %</div><div class="monitor-value" id="monMem">-</div></div>
                <div class="monitor-metric"><div class="monitor-label">Elapsed</div><div class="monitor-value" id="monElapsed">-</div></div>
                <div class="monitor-metric"><div class="monitor-label">Snapshot UTC</div><div class="monitor-value" id="monSnapshot">-</div></div>
            </div>
            <div class="trend-grid">
                <div class="trend-card">
                    <div class="trend-title">Rows/Min (10m) Trend</div>
                    <svg id="trendRpm10" class="trend-svg" viewBox="0 0 320 90" preserveAspectRatio="none"></svg>
                    <div class="trend-caption" id="trendRpm10Caption">No trend data yet.</div>
                </div>
                <div class="trend-card">
                    <div class="trend-title">Rows Total Trend</div>
                    <svg id="trendRowsTotal" class="trend-svg" viewBox="0 0 320 90" preserveAspectRatio="none"></svg>
                    <div class="trend-caption" id="trendRowsTotalCaption">No trend data yet.</div>
                </div>
            </div>
            <p class="info">Read-only metrics from logs + DuckDB, safe while long runs are active.</p>
        </div>
        
        <div class="card card-wide">
            <h2>Scan</h2>
            <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
                <button class="btn btn-primary" onclick="runScan()" id="scanBtn">🔍 Run Now</button>
                <button class="btn btn-danger" onclick="stopOp('scan')" id="stopScanBtn" style="display:none;">🛑 Stop Scan</button>
                <span id="scanStatus" style="color:#888; font-size:0.85rem;"></span>
            </div>
            <div class="schedule-form" style="margin-top:15px;">
                <div class="schedule-row">
                    <input type="time" id="scheduleTime" value="09:30">
                    <label style="color:#888; display:flex; align-items:center; gap:5px;">
                        <input type="checkbox" id="scanRepeatCheck" onchange="toggleScanRepeat()">
                        Also every
                    </label>
                    <input type="number" id="scanIntervalHours" value="12" min="1" max="23" style="width:50px;" disabled>
                    <span style="color:#888;">hours</span>
                </div>
                <div class="schedule-row">
                    <button class="btn btn-warning" onclick="setSchedule()" id="scheduleBtn">⏰ Schedule</button>
                    <button class="btn btn-danger" onclick="cancelSchedule()" id="cancelBtn" style="display:none;">✖ Cancel</button>
                </div>
                <div class="days-selector" id="scanDays">
                    <label class="day-check"><input type="checkbox" value="0" checked><span>M</span></label>
                    <label class="day-check"><input type="checkbox" value="1" checked><span>T</span></label>
                    <label class="day-check"><input type="checkbox" value="2" checked><span>W</span></label>
                    <label class="day-check"><input type="checkbox" value="3" checked><span>R</span></label>
                    <label class="day-check"><input type="checkbox" value="4" checked><span>F</span></label>
                    <label class="day-check"><input type="checkbox" value="5"><span>S</span></label>
                    <label class="day-check"><input type="checkbox" value="6"><span>N</span></label>
                </div>
            </div>
            <p class="info">📅 Runs at scheduled time on selected days. Optional: repeat every N hours.</p>
            <div id="scheduleStatus"></div>
        </div>
        
        <div class="card card-compact">
            <h2>Training</h2>
            <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
                <button class="btn btn-secondary" onclick="runTrain()" id="trainBtn">🧠 Run Now</button>
                <button class="btn btn-danger" onclick="stopOp('train')" id="stopTrainBtn" style="display:none;">🛑 Stop Train</button>
                <span id="trainStatus" style="color:#888; font-size:0.85rem;"></span>
            </div>
            <div class="schedule-form" style="margin-top:15px;">
                <div class="schedule-row">
                    <input type="time" id="trainTime" value="21:00">
                    <span style="color:#888;">every</span>
                    <input type="number" id="trainDays" value="10" min="1" max="90" style="width:60px;">
                    <span style="color:#888;">days</span>
                </div>
                <div class="schedule-row">
                    <button class="btn btn-warning" onclick="setTrainSchedule()" id="trainScheduleBtn">⏰ Schedule</button>
                    <button class="btn btn-danger" onclick="cancelTrainSchedule()" id="trainCancelBtn" style="display:none;">✖ Cancel</button>
                </div>
                <div class="days-selector" id="trainDayChecks">
                    <label class="day-check"><input type="checkbox" value="0" checked><span>M</span></label>
                    <label class="day-check"><input type="checkbox" value="1" checked><span>T</span></label>
                    <label class="day-check"><input type="checkbox" value="2" checked><span>W</span></label>
                    <label class="day-check"><input type="checkbox" value="3" checked><span>R</span></label>
                    <label class="day-check"><input type="checkbox" value="4" checked><span>F</span></label>
                    <label class="day-check"><input type="checkbox" value="5" checked><span>S</span></label>
                    <label class="day-check"><input type="checkbox" value="6" checked><span>N</span></label>
                </div>
            </div>
            <p class="info">🔄 Repeats every N days on selected weekdays.</p>
            <div id="trainScheduleStatus"></div>
        </div>
        
        <div class="card card-compact">
            <h2>Qualitative Features</h2>
            <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
                <button class="btn btn-primary" onclick="runQual()" id="qualBtn">🏭 Run Now</button>
                <button class="btn btn-danger" onclick="stopOp('qual')" id="stopQualBtn" style="display:none;">🛑 Stop Qual</button>
                <span id="qualOpStatus" style="color:#888; font-size:0.85rem;"></span>
            </div>
            <div class="schedule-form" style="margin-top:15px;">
                <div class="schedule-row">
                    <label style="color:#888;">Chunk:</label>
                    <input type="number" id="qualChunk" value="20" min="5" max="100" style="width:60px;">
                    <label style="color:#888;">Interval:</label>
                    <input type="number" id="qualInterval" value="4" min="1" max="24" style="width:50px;">
                    <span style="color:#888;">hours</span>
                </div>
                <div class="schedule-row">
                    <button class="btn btn-warning" onclick="setQualSchedule()" id="qualScheduleBtn">⏰ Schedule</button>
                    <button class="btn btn-danger" onclick="cancelQualSchedule()" id="qualCancelBtn" style="display:none;">✖ Cancel</button>
                </div>
            </div>
            <p class="info">🔄 Industry/sector data via gemini-cli. One ticker at a time.</p>
            <div id="qualStatus"></div>
        </div>

        <div class="card card-compact">
            <h2>News Assessment</h2>
            <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
                <button class="btn btn-primary" onclick="runNews()" id="newsBtn">📰 Run Now</button>
                <button class="btn btn-danger" onclick="stopOp('news')" id="stopNewsBtn" style="display:none;">🛑 Stop News</button>
                <span id="newsOpStatus" style="color:#888; font-size:0.85rem;"></span>
            </div>
            <div class="schedule-form" style="margin-top:15px;">
                <div class="schedule-row">
                    <label style="color:#888;">Morning:</label>
                    <input type="time" id="newsMorningTime" value="08:30">
                    <label style="color:#888;">Evening:</label>
                    <input type="time" id="newsEveningTime" value="16:30">
                </div>
                <div class="schedule-row">
                    <button class="btn btn-warning" onclick="setNewsSchedule()" id="newsScheduleBtn">⏰ Schedule</button>
                    <button class="btn btn-danger" onclick="cancelNewsSchedule()" id="newsCancelBtn" style="display:none;">✖ Cancel</button>
                </div>
                <div class="days-selector" id="newsDays">
                    <label class="day-check"><input type="checkbox" value="0" checked><span>M</span></label>
                    <label class="day-check"><input type="checkbox" value="1" checked><span>T</span></label>
                    <label class="day-check"><input type="checkbox" value="2" checked><span>W</span></label>
                    <label class="day-check"><input type="checkbox" value="3" checked><span>R</span></label>
                    <label class="day-check"><input type="checkbox" value="4" checked><span>F</span></label>
                    <label class="day-check"><input type="checkbox" value="5"><span>S</span></label>
                    <label class="day-check"><input type="checkbox" value="6"><span>N</span></label>
                </div>
            </div>
            <p class="info">📰 Trade policy, geopolitical, regulatory & event shock scores via Gemini CLI (morning pre-market + evening post-close).</p>
            <div id="newsStatus"></div>
        </div>

        <div class="card card-compact">
            <h2>Audit</h2>
            <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
                <button class="btn btn-secondary" onclick="runAudit()" id="auditBtn">📋 Run Now</button>
                <button class="btn btn-danger" onclick="stopOp('audit')" id="stopAuditBtn" style="display:none;">🛑 Stop Audit</button>
                <span id="auditOpStatus" style="color:#888; font-size:0.85rem;"></span>
            </div>
            <div class="schedule-form" style="margin-top:15px;">
                <div class="schedule-row">
                    <input type="time" id="auditTime" value="18:30">
                </div>
                <div class="schedule-row">
                    <button class="btn btn-warning" onclick="setAuditSchedule()" id="auditScheduleBtn">⏰ Schedule</button>
                    <button class="btn btn-danger" onclick="cancelAuditSchedule()" id="auditCancelBtn" style="display:none;">✖ Cancel</button>
                </div>
                <div class="days-selector" id="auditDays">
                    <label class="day-check"><input type="checkbox" value="0" checked><span>M</span></label>
                    <label class="day-check"><input type="checkbox" value="1" checked><span>T</span></label>
                    <label class="day-check"><input type="checkbox" value="2" checked><span>W</span></label>
                    <label class="day-check"><input type="checkbox" value="3" checked><span>R</span></label>
                    <label class="day-check"><input type="checkbox" value="4" checked><span>F</span></label>
                    <label class="day-check"><input type="checkbox" value="5"><span>S</span></label>
                    <label class="day-check"><input type="checkbox" value="6"><span>N</span></label>
                </div>
            </div>
            <p class="info">Runs `audit + notify` and refreshes public performance outputs on selected days.</p>
            <div id="auditStatus"></div>
        </div>

        <div class="card card-compact">
            <h2>Weekly Summary</h2>
            <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
                <button class="btn btn-secondary" onclick="runWeeklySummary()" id="weeklySummaryBtn">🗞️ Run Now</button>
                <span id="weeklySummaryOpStatus" style="color:#888; font-size:0.85rem;"></span>
            </div>
            <div class="schedule-form" style="margin-top:15px;">
                <div class="schedule-row">
                    <label style="color:#888;">Day:</label>
                    <select id="weeklySummaryDay" style="padding:9px 10px; border-radius:8px; border:1px solid #ced9e5; background:#fff; color:#102337;">
                        <option value="0">Monday</option>
                        <option value="1">Tuesday</option>
                        <option value="2">Wednesday</option>
                        <option value="3">Thursday</option>
                        <option value="4" selected>Friday</option>
                        <option value="5">Saturday</option>
                        <option value="6">Sunday</option>
                    </select>
                    <input type="time" id="weeklySummaryTime" value="18:45">
                </div>
                <div class="schedule-row">
                    <button class="btn btn-warning" onclick="setWeeklySummarySchedule()" id="weeklySummaryScheduleBtn">⏰ Schedule</button>
                    <button class="btn btn-danger" onclick="cancelWeeklySummarySchedule()" id="weeklySummaryCancelBtn" style="display:none;">✖ Cancel</button>
                </div>
            </div>
            <p class="info">Sends a compact weekly performance notification from latest public audit output.</p>
            <div id="weeklySummaryStatus"></div>
        </div>

        <div class="card card-compact">
            <h2>Model Performance</h2>
            <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
                <a class="btn btn-secondary" href="/public/performance" target="_blank" rel="noopener">📈 Open Public Dashboard</a>
            </div>
            <p class="info">Updated by scheduled daily run (`scan + audit + report + notify`).</p>
        </div>

        <div class="card card-wide">
            <h2>Portfolio & Config</h2>
            <div class="field-grid">
                <div class="field" style="grid-column: span 2;">
                    <label for="portfolioHoldings">Portfolio Holdings (comma/newline)</label>
                    <textarea id="portfolioHoldings" placeholder="AAPL, MSFT, SPY"></textarea>
                </div>
                <div class="field">
                    <label for="watchlistInput">Watchlist</label>
                    <textarea id="watchlistInput" placeholder="NVDA, AMZN"></textarea>
                </div>
                <div class="field">
                    <label for="denylistInput">Universe Denylist</label>
                    <textarea id="denylistInput" placeholder="TICKERS_TO_EXCLUDE"></textarea>
                </div>
                <div class="field" style="grid-column: span 2;">
                    <label for="purgeTickers">Quick Purge (adds to denylist)</label>
                    <div class="schedule-row">
                        <input type="text" id="purgeTickers" placeholder="TICK1, TICK2, TICK3">
                        <button class="btn btn-danger" onclick="purgeUniverseTickers()">🗑 Purge</button>
                        <button class="btn btn-secondary" onclick="loadPurgeSuggestions()">💡 Suggest Purge</button>
                        <button class="btn btn-warning" onclick="purgeSelectedSuggestions()">✅ Purge Selected</button>
                    </div>
                    <div id="purgeStatus" class="info"></div>
                    <div id="purgeSuggestStatus" class="info"></div>
                    <div id="purgeSuggestions" class="history-list"></div>
                </div>
                <div class="field">
                    <label for="confluenceThreshold">Confluence Threshold</label>
                    <input type="number" step="0.01" min="0" max="1" id="confluenceThreshold">
                </div>
                <div class="field">
                    <label for="tacticalThreshold">Tactical Threshold</label>
                    <input type="number" step="0.01" min="0" max="1" id="tacticalThreshold">
                </div>
                <div class="field">
                    <label for="trendThreshold">Trend Threshold</label>
                    <input type="number" step="0.01" min="0" max="1" id="trendThreshold">
                </div>
                <div class="field">
                    <label for="tradabilityMinPrice">Tradability Min Price</label>
                    <input type="number" step="0.01" min="0" id="tradabilityMinPrice">
                </div>
                <div class="field">
                    <label for="tradabilityMinVolume">Tradability Min Avg Vol 20d</label>
                    <input type="number" step="1" min="0" id="tradabilityMinVolume">
                </div>
                <div class="field">
                    <label for="tradabilityMinDollarVolume">Tradability Min Avg Dollar Vol 20d</label>
                    <input type="number" step="1" min="0" id="tradabilityMinDollarVolume">
                </div>
                <div class="field">
                    <label for="tradabilityMaxAtrRatio">Tradability Max ATR Ratio</label>
                    <input type="number" step="0.001" min="0" id="tradabilityMaxAtrRatio">
                </div>
                <div class="field">
                    <label for="horizonTarget5">Target Return 5d</label>
                    <input type="number" step="0.001" id="horizonTarget5">
                </div>
                <div class="field">
                    <label for="horizonTarget10">Target Return 10d</label>
                    <input type="number" step="0.001" id="horizonTarget10">
                </div>
                <div class="field">
                    <label for="horizonTarget30">Target Return 30d</label>
                    <input type="number" step="0.001" id="horizonTarget30">
                </div>
            </div>
            <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:12px;">
                <button class="btn btn-secondary" onclick="loadConfigEditor()">↻ Load Config</button>
                <button class="btn btn-primary" onclick="saveConfigEditor()">💾 Save Config</button>
            </div>
            <div id="configSaveStatus" class="info"></div>
            <div class="schedule-row" style="margin-top:10px;">
                <button class="btn btn-secondary" onclick="loadConfigHistory()">⏱ Config History</button>
            </div>
            <div id="configHistory" class="history-list"></div>
        </div>

        <div class="card card-full">
            <h2>Reports & Logs</h2>
            <div class="schedule-row">
                <button class="btn btn-secondary" onclick="loadReportFiles()">📄 Refresh Reports</button>
                <button class="btn btn-secondary" onclick="loadLogFiles()">🧾 Refresh Logs</button>
            </div>
            <div class="field-grid" style="margin-top:8px;">
                <div class="field">
                    <label>Recent Reports</label>
                    <div id="reportFiles" class="file-list"></div>
                </div>
                <div class="field">
                    <label>Inline Preview</label>
                    <div class="preview-shell">
                        <div class="preview-header" id="previewTitle">Select a report/log file</div>
                        <pre id="previewContent" class="preview-content"></pre>
                    </div>
                </div>
                <div class="field" style="grid-column: span 2;">
                    <label>Recent Logs</label>
                    <div id="logFiles" class="file-list"></div>
                </div>
            </div>
            <p class="info">Use Preview for quick triage; Open launches full file in a new tab.</p>
        </div>
        </div>
    </div>
    
    <script>
        let serverTimeOffset = 0;
        const DAYS = ['M','T','W','R','F','S','N'];
        let configLoaded = false;
        let reportCount = 0;
        let logCount = 0;

        function basename(path) {
            if (!path) return '';
            const bits = String(path).split('/');
            return bits[bits.length - 1] || '';
        }

        function toLocal(isoLike) {
            if (!isoLike) return '-';
            const dt = new Date(isoLike);
            if (Number.isNaN(dt.getTime())) return isoLike;
            return dt.toLocaleString();
        }

        function setSummaryTile(valueId, noteId, value, note) {
            const valueEl = document.getElementById(valueId);
            const noteEl = document.getElementById(noteId);
            if (valueEl) valueEl.textContent = value;
            if (noteEl) noteEl.textContent = note;
        }

        function normalizedSeries(values) {
            const out = (values || []).map(v => Number(v)).filter(v => Number.isFinite(v));
            return out;
        }

        function renderSparkline(svgId, values, color='#0ea5e9') {
            const svg = document.getElementById(svgId);
            if (!svg) return;
            const nums = normalizedSeries(values);
            if (nums.length < 2) {
                svg.innerHTML = '';
                return;
            }
            const width = 320;
            const height = 90;
            const pad = 6;
            const min = Math.min(...nums);
            const max = Math.max(...nums);
            const span = Math.max(1e-9, max - min);
            const xStep = (width - (pad * 2)) / Math.max(1, nums.length - 1);
            const points = nums.map((v, idx) => {
                const x = pad + (idx * xStep);
                const yNorm = (v - min) / span;
                const y = (height - pad) - (yNorm * (height - (pad * 2)));
                return `${x.toFixed(2)},${y.toFixed(2)}`;
            });
            const polyline = `<polyline points="${points.join(' ')}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"></polyline>`;
            const first = points[0].split(',');
            const last = points[points.length - 1].split(',');
            const area = `<polygon points="${first[0]},${height - pad} ${points.join(' ')} ${last[0]},${height - pad}" fill="${color}" opacity="0.10"></polygon>`;
            svg.innerHTML = area + polyline;
        }

        function renderMonitorTrends(history) {
            const items = Array.isArray(history) ? history.slice().reverse() : [];
            const rpmSeries = items.map(item => {
                const r = item.rows_per_min || {};
                return Number(r['10m'] ?? r['15m'] ?? r['5m'] ?? r['20m'] ?? r['60m'] ?? 0);
            });
            const rowsSeries = items.map(item => Number(item.rows_total || 0));
            renderSparkline('trendRpm10', rpmSeries, '#0ea5e9');
            renderSparkline('trendRowsTotal', rowsSeries, '#0f8b8d');

            const rpmNow = rpmSeries.length ? rpmSeries[rpmSeries.length - 1] : null;
            const rowsNow = rowsSeries.length ? rowsSeries[rowsSeries.length - 1] : null;
            const rpmText = (rpmNow == null) ? 'No trend data yet.' : `${items.length} snapshots · latest ${rpmNow.toFixed(1)} rows/min`;
            const rowsText = (rowsNow == null) ? 'No trend data yet.' : `${items.length} snapshots · latest ${rowsNow.toLocaleString()} rows`;
            const rpmEl = document.getElementById('trendRpm10Caption');
            const rowsEl = document.getElementById('trendRowsTotalCaption');
            if (rpmEl) rpmEl.textContent = rpmText;
            if (rowsEl) rowsEl.textContent = rowsText;
        }

        function getSelectedDays(elementId) {
            const container = document.getElementById(elementId);
            const checkboxes = container.querySelectorAll('input[type="checkbox"]');
            const selected = [];
            checkboxes.forEach(cb => {
                if (cb.checked) selected.push(parseInt(cb.value));
            });
            return selected;
        }

        function formatDays(dayList) {
            if (!dayList || dayList.length === 7) return "All";
            if (dayList.length === 0) return "None";
            const isWeekdays = dayList.length === 5 && [0,1,2,3,4].every(d => dayList.includes(d));
            if (isWeekdays) return "M-F";
            return dayList.map(d => DAYS[d]).join('');
        }

        function parseTickerText(raw) {
            return (raw || '')
                .split(/[\\s,;]+/)
                .map(x => x.trim().toUpperCase())
                .filter(Boolean);
        }

        function setTickerTextarea(id, values) {
            const list = Array.isArray(values) ? values : [];
            document.getElementById(id).value = list.join(', ');
        }

        function setStatusText(id, text, isError=false) {
            const el = document.getElementById(id);
            if (!el) return;
            el.textContent = text || '';
            el.style.color = isError ? '#ef4444' : '#9ca3af';
        }

        function setPreviewText(title, content) {
            const titleEl = document.getElementById('previewTitle');
            const contentEl = document.getElementById('previewContent');
            if (titleEl) titleEl.textContent = title || 'Select a report/log file';
            if (contentEl) contentEl.textContent = content || '';
        }

        function previewFile(kind, relPath) {
            if (!relPath) return;
            fetch(`/api/file_preview?kind=${encodeURIComponent(kind)}&path=${encodeURIComponent(relPath)}&max_chars=45000`)
                .then(r => r.json())
                .then(data => {
                    if (data.status !== 'ok') {
                        setPreviewText(`Preview failed: ${relPath}`, data.message || 'Unknown error');
                        return;
                    }
                    const tag = data.truncated ? ' (truncated)' : '';
                    setPreviewText(`${data.kind}/${data.rel_path} · ${data.mtime}${tag}`, data.content || '');
                })
                .catch(err => setPreviewText(`Preview failed: ${relPath}`, String(err)));
        }

        function loadConfigEditor() {
            fetch('/api/config')
                .then(r => r.json())
                .then(data => {
                    if (data.status !== 'ok') {
                        setStatusText('configSaveStatus', data.message || 'Failed to load config', true);
                        return;
                    }
                    const cfg = data.config || {};
                    setTickerTextarea('portfolioHoldings', cfg.portfolio_holdings || []);
                    setTickerTextarea('watchlistInput', cfg.watchlist || []);
                    setTickerTextarea('denylistInput', cfg.universe_denylist || []);

                    document.getElementById('confluenceThreshold').value = cfg.confluence_threshold ?? 0.60;
                    document.getElementById('tacticalThreshold').value = cfg.tactical_threshold ?? 0.70;
                    document.getElementById('trendThreshold').value = cfg.trend_threshold ?? 0.65;
                    document.getElementById('tradabilityMinPrice').value = cfg.tradability_min_price ?? 3.0;
                    document.getElementById('tradabilityMinVolume').value = cfg.tradability_min_avg_volume_20d ?? 300000;
                    document.getElementById('tradabilityMinDollarVolume').value = cfg.tradability_min_avg_dollar_volume_20d ?? 5000000;
                    document.getElementById('tradabilityMaxAtrRatio').value = cfg.tradability_max_atr_ratio ?? 0.20;
                    const h = cfg.horizon_targets || {};
                    document.getElementById('horizonTarget5').value = h['5'] ?? 0.015;
                    document.getElementById('horizonTarget10').value = h['10'] ?? 0.030;
                    document.getElementById('horizonTarget30').value = h['30'] ?? 0.080;
                    configLoaded = true;
                    setStatusText('configSaveStatus', 'Config loaded.');
                    loadPurgeSuggestions();
                })
                .catch(err => setStatusText('configSaveStatus', `Load failed: ${err}`, true));
        }

        function saveConfigEditor() {
            const payload = {
                portfolio_holdings: parseTickerText(document.getElementById('portfolioHoldings').value),
                watchlist: parseTickerText(document.getElementById('watchlistInput').value),
                universe_denylist: parseTickerText(document.getElementById('denylistInput').value),
                confluence_threshold: parseFloat(document.getElementById('confluenceThreshold').value),
                tactical_threshold: parseFloat(document.getElementById('tacticalThreshold').value),
                trend_threshold: parseFloat(document.getElementById('trendThreshold').value),
                tradability_min_price: parseFloat(document.getElementById('tradabilityMinPrice').value),
                tradability_min_avg_volume_20d: parseInt(document.getElementById('tradabilityMinVolume').value),
                tradability_min_avg_dollar_volume_20d: parseInt(document.getElementById('tradabilityMinDollarVolume').value),
                tradability_max_atr_ratio: parseFloat(document.getElementById('tradabilityMaxAtrRatio').value),
                horizon_targets: {
                    "5": parseFloat(document.getElementById('horizonTarget5').value),
                    "10": parseFloat(document.getElementById('horizonTarget10').value),
                    "30": parseFloat(document.getElementById('horizonTarget30').value),
                },
            };
            fetch('/api/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload),
            })
                .then(r => r.json())
                .then(data => {
                    if (data.status !== 'ok') {
                        setStatusText('configSaveStatus', data.message || 'Save failed', true);
                        return;
                    }
                    const keys = (data.updated_keys || []).join(', ');
                    setStatusText('configSaveStatus', `Saved: ${keys}`);
                    loadConfigEditor();
                    updateStatus();
                })
                .catch(err => setStatusText('configSaveStatus', `Save failed: ${err}`, true));
        }

        function purgeUniverseTickers() {
            const raw = document.getElementById('purgeTickers').value || '';
            const tickers = parseTickerText(raw);
            if (!tickers.length) {
                setStatusText('purgeStatus', 'Enter one or more tickers.', true);
                return;
            }
            fetch('/api/universe/purge', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({tickers: tickers}),
            })
                .then(r => r.json())
                .then(data => {
                    if (data.status !== 'ok') {
                        setStatusText('purgeStatus', data.message || 'Purge failed', true);
                        return;
                    }
                    const added = (data.added || []).join(', ') || 'none';
                    setStatusText('purgeStatus', `Purged: ${added}. Denylist size: ${data.total_denylist || 0}`);
                    document.getElementById('purgeTickers').value = '';
                    loadConfigEditor();
                    loadConfigHistory();
                    updateStatus();
                    loadPurgeSuggestions();
                })
                .catch(err => setStatusText('purgeStatus', `Purge failed: ${err}`, true));
        }

        function renderPurgeSuggestions(items) {
            const container = document.getElementById('purgeSuggestions');
            if (!container) return;
            if (!Array.isArray(items) || items.length === 0) {
                container.innerHTML = '<div class="history-item"><div>No candidates right now.</div></div>';
                return;
            }
            container.innerHTML = items.map(item => {
                const ticker = item.ticker || '';
                const reasons = Array.isArray(item.reasons) ? item.reasons.join(', ') : '';
                const vol = (item.avg_volume_20d == null) ? '-' : Number(item.avg_volume_20d).toLocaleString();
                const dol = (item.avg_dollar_volume_20d == null) ? '-' : Number(item.avg_dollar_volume_20d).toLocaleString();
                const lastDate = item.last_date || '-';
                const staleDays = (item.stale_days == null) ? '-' : `${item.stale_days}d`;
                const preds = (item.pred_rows_60d == null) ? '-' : `${item.pred_rows_60d}`;
                return `
                    <div class="history-item">
                        <div>
                            <div><label><input type="checkbox" class="purge-suggest-cb" value="${ticker}"> <strong>${ticker}</strong></label></div>
                            <div class="history-meta">score=${item.score || 0} · reasons: ${reasons}</div>
                            <div class="history-meta">last=${lastDate} (${staleDays}) · vol20=${vol} · $vol20=${dol} · preds60=${preds}</div>
                        </div>
                        <div><button class="btn btn-danger" onclick="purgeUniverseTickersDirect(['${ticker}'])">Purge</button></div>
                    </div>
                `;
            }).join('');
        }

        function purgeUniverseTickersDirect(tickers) {
            const clean = (Array.isArray(tickers) ? tickers : []).map(t => String(t || '').trim().toUpperCase()).filter(Boolean);
            if (!clean.length) {
                setStatusText('purgeStatus', 'No tickers selected.', true);
                return;
            }
            fetch('/api/universe/purge', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({tickers: clean}),
            })
                .then(r => r.json())
                .then(data => {
                    if (data.status !== 'ok' && data.status !== 'noop') {
                        setStatusText('purgeStatus', data.message || 'Purge failed', true);
                        return;
                    }
                    const added = (data.added || []).join(', ') || 'none';
                    setStatusText('purgeStatus', `Purged: ${added}. Denylist size: ${data.total_denylist || 0}`);
                    loadConfigEditor();
                    loadConfigHistory();
                    updateStatus();
                    loadPurgeSuggestions();
                })
                .catch(err => setStatusText('purgeStatus', `Purge failed: ${err}`, true));
        }

        function purgeSelectedSuggestions() {
            const selected = [];
            document.querySelectorAll('.purge-suggest-cb').forEach(cb => {
                if (cb.checked) selected.push(String(cb.value || '').trim().toUpperCase());
            });
            purgeUniverseTickersDirect(selected);
        }

        function loadPurgeSuggestions() {
            fetch('/api/universe/purge_suggestions?limit=25')
                .then(r => {
                    if (r.status === 404) {
                        return fetch('/api/universe/purge-suggestions?limit=25');
                    }
                    return r;
                })
                .then(r => r.json())
                .then(data => {
                    if (data.status !== 'ok') {
                        setStatusText('purgeSuggestStatus', data.message || 'Failed to load suggestions', true);
                        renderPurgeSuggestions([]);
                        return;
                    }
                    const count = (data.suggestions || []).length;
                    const total = data.total_candidates ?? count;
                    setStatusText('purgeSuggestStatus', `Loaded ${count}/${total} suggestions.`);
                    renderPurgeSuggestions(data.suggestions || []);
                })
                .catch(err => {
                    setStatusText('purgeSuggestStatus', `Suggest failed: ${err}`, true);
                    renderPurgeSuggestions([]);
                });
        }

        function loadConfigHistory() {
            fetch('/api/config/history?limit=30')
                .then(r => r.json())
                .then(data => {
                    const container = document.getElementById('configHistory');
                    if (!container) return;
                    const items = data.items || [];
                    if (!items.length) {
                        container.innerHTML = '<div class="history-item"><div>No config history yet.</div></div>';
                        return;
                    }
                    container.innerHTML = items.map(item => {
                        const id = item.id || '';
                        const ts = item.ts || '';
                        const keys = (item.updated_keys || []).join(', ') || 'unknown';
                        const action = item.action || 'update';
                        return `
                            <div class="history-item">
                                <div>
                                    <div><strong>${ts}</strong> · ${action}</div>
                                    <div class="history-meta">id=${id} · keys: ${keys}</div>
                                </div>
                                <div><button class="btn btn-warning" onclick="rollbackConfig('${id}')" style="min-width:100px;">Rollback</button></div>
                            </div>
                        `;
                    }).join('');
                })
                .catch(() => {
                    const container = document.getElementById('configHistory');
                    if (container) container.innerHTML = '<div class="history-item"><div>Failed to load history.</div></div>';
                });
        }

        function rollbackConfig(historyId) {
            if (!historyId) return;
            fetch('/api/config/rollback', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({id: historyId}),
            })
                .then(r => r.json())
                .then(data => {
                    if (data.status !== 'ok') {
                        setStatusText('configSaveStatus', data.message || 'Rollback failed', true);
                        return;
                    }
                    const keys = (data.updated_keys || []).join(', ');
                    setStatusText('configSaveStatus', `Rollback complete: ${keys || 'updated'}`);
                    loadConfigEditor();
                    loadConfigHistory();
                    updateStatus();
                })
                .catch(err => setStatusText('configSaveStatus', `Rollback failed: ${err}`, true));
        }

        function loadRunMonitor() {
            fetch('/api/run_monitor')
                .then(r => r.json())
                .then(data => {
                    if (data.status !== 'ok') {
                        setStatusText('runMonitorStatus', data.message || 'Failed to load run monitor', true);
                        return;
                    }
                    const mon = data.monitor || {};
                    const db = mon.db || {};
                    const rp = db.replay_progress || {};
                    const proc = mon.process || {};
                    const rates = db.rows_per_min || {};
                    const rpm10 = Number(rates['10m'] ?? rates['15m'] ?? rates['5m'] ?? 0);
                    const hash = db.pred_latest_hash || '';
                    const shortHash = hash ? `${hash.slice(0, 10)}...` : '-';
                    const cpu = (proc.cpu_percent == null) ? null : Number(proc.cpu_percent);
                    const mem = (proc.mem_percent == null) ? null : Number(proc.mem_percent);
                    const elapsed = proc.elapsed || '-';
                    const snapshotUtc = db.snapshot_generated_at_utc || '';
                    const rowsDone = Number(rp.rows_done || db.pred_rows_total || 0);

                    document.getElementById('monPhase').textContent = mon.phase || '-';
                    document.getElementById('monProc').textContent = mon.process_running
                        ? `RUNNING${proc.pid ? ` #${proc.pid}` : ''}`
                        : 'IDLE';
                    document.getElementById('monRpm10').textContent = Number(rpm10).toFixed(1);
                    document.getElementById('monEta').textContent = (rp.eta_minutes == null) ? '-' : `${rp.eta_minutes} min`;
                    document.getElementById('monProgress').textContent = (rp.progress_pct == null) ? '-' : `${rp.progress_pct}%`;
                    document.getElementById('monRowsDone').textContent = rowsDone.toLocaleString();
                    document.getElementById('monAsOf').textContent = rp.max_asof_date || '-';
                    document.getElementById('monHash').textContent = shortHash;
                    document.getElementById('monCpu').textContent = (cpu == null) ? '-' : `${cpu.toFixed(1)}%`;
                    document.getElementById('monMem').textContent = (mem == null) ? '-' : `${mem.toFixed(1)}%`;
                    document.getElementById('monElapsed').textContent = elapsed;
                    document.getElementById('monSnapshot').textContent = snapshotUtc || '-';

                    const full = mon.full_log || {};
                    const post = mon.post_log || {};
                    const fullStatus = full.done ? 'full:done' : 'full:running';
                    const postStatus = post.done ? 'post:done' : 'post:waiting';
                    const rcBits = [];
                    if (post.full_run_rc != null) rcBits.push(`full_run_rc=${post.full_run_rc}`);
                    if (post.integration_rc != null) rcBits.push(`integration_rc=${post.integration_rc}`);
                    if (post.runtime_skew_rc != null) rcBits.push(`runtime_skew_rc=${post.runtime_skew_rc}`);
                    const rcText = rcBits.length ? ` · ${rcBits.join(' · ')}` : '';
                    let utilHint = '';
                    if (mon.process_running && cpu != null && cpu < 25) utilHint = ' · low CPU (likely I/O or wait phase)';
                    setStatusText('runMonitorStatus', `${mon.now || ''} · ${fullStatus} · ${postStatus}${rcText}${utilHint}`);

                    const fullLink = document.getElementById('fullLogLink');
                    const postLink = document.getElementById('postLogLink');
                    const fullName = basename(full.path || '');
                    const postName = basename(post.path || '');
                    if (fullLink) {
                        if (fullName) {
                            fullLink.href = `/files/open?kind=logs&path=${encodeURIComponent(fullName)}`;
                            fullLink.textContent = `Open ${fullName}`;
                            fullLink.style.pointerEvents = 'auto';
                            fullLink.style.opacity = '1';
                        } else {
                            fullLink.href = '#';
                            fullLink.textContent = 'Open Full Log';
                            fullLink.style.pointerEvents = 'none';
                            fullLink.style.opacity = '0.6';
                        }
                    }
                    if (postLink) {
                        if (postName) {
                            postLink.href = `/files/open?kind=logs&path=${encodeURIComponent(postName)}`;
                            postLink.textContent = `Open ${postName}`;
                            postLink.style.pointerEvents = 'auto';
                            postLink.style.opacity = '1';
                        } else {
                            postLink.href = '#';
                            postLink.textContent = 'Open Post-Check';
                            postLink.style.pointerEvents = 'none';
                            postLink.style.opacity = '0.6';
                        }
                    }

                    const procSummaryValue = mon.process_running
                        ? ((cpu == null) ? 'Running' : `${cpu.toFixed(1)}% CPU`)
                        : 'Idle';
                    const procSummaryNote = mon.process_running
                        ? `pid ${proc.pid || '-'} · elapsed ${elapsed}`
                        : 'No active process';
                    setSummaryTile('sumProc', 'sumProcNote', procSummaryValue, procSummaryNote);

                    const snapSummaryValue = snapshotUtc ? toLocal(snapshotUtc) : '-';
                    const snapSummaryNote = snapshotUtc
                        ? `${rowsDone.toLocaleString()} rows · reports ${reportCount} · logs ${logCount}`
                        : 'No snapshot yet';
                    setSummaryTile('sumSnapshot', 'sumSnapshotNote', snapSummaryValue, snapSummaryNote);
                    renderMonitorTrends(mon.snapshot_history || []);
                })
                .catch(err => setStatusText('runMonitorStatus', `Run monitor failed: ${err}`, true));
        }
        
        function renderFileRows(containerId, files, kind) {
            const container = document.getElementById(containerId);
            if (!container) return;
            if (!Array.isArray(files) || files.length === 0) {
                container.innerHTML = '<div class="file-row"><div>No files found.</div></div>';
                return;
            }
            container.innerHTML = files.map(file => {
                const rel = file.rel_path || file.name;
                const href = `/files/open?kind=${encodeURIComponent(kind)}&path=${encodeURIComponent(rel)}`;
                const encodedRel = encodeURIComponent(rel);
                const previewCall = `previewFile('${kind}', decodeURIComponent('${encodedRel}'))`;
                return `
                    <div class="file-row">
                        <div>
                            <div>${rel}</div>
                            <div class="file-meta">${file.mtime || ''} · ${file.size || 0} bytes</div>
                        </div>
                        <div class="file-actions">
                            <a href="${href}" target="_blank" rel="noopener">Open</a>
                            <button onclick="${previewCall}">Preview</button>
                        </div>
                    </div>
                `;
            }).join('');
        }

        function loadReportFiles() {
            fetch('/api/reports')
                .then(r => r.json())
                .then(data => {
                    const files = data.files || [];
                    reportCount = files.length;
                    renderFileRows('reportFiles', files, 'reports');
                })
                .catch(() => renderFileRows('reportFiles', [], 'reports'));
        }

        function loadLogFiles() {
            fetch('/api/logs')
                .then(r => r.json())
                .then(data => {
                    const files = data.files || [];
                    logCount = files.length;
                    renderFileRows('logFiles', files, 'logs');
                })
                .catch(() => renderFileRows('logFiles', [], 'logs'));
        }
        
        function updateClock() {
            const serverNow = new Date(Date.now() + serverTimeOffset);
            document.getElementById('clock').textContent = serverNow.toLocaleTimeString() + ' (Server)';
        }
        setInterval(updateClock, 1000);
        updateClock();
        
        function toggleScanRepeat() {
            const check = document.getElementById('scanRepeatCheck');
            document.getElementById('scanIntervalHours').disabled = !check.checked;
        }
        
        function updateStatus() {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    // Running ops summary
                    const runningOps = document.getElementById('runningOps');
                    let runningList = [];
                    if (data.scan_running) runningList.push('🔍 Scan');
                    if (data.train_running) runningList.push('🧠 Train');
                    if (data.qual_running) runningList.push('🏭 Qual');
                    if (data.news_running) runningList.push('📰 News');
                    if (data.audit_running) runningList.push('📋 Audit');
                    if (data.weekly_summary_running) runningList.push('🗞️ Weekly');
                    if (data.snapshot_running) runningList.push('📸 Snapshot');
                    
                    if (runningList.length > 0) {
                        runningOps.innerHTML = `<div class="status status-running running">${runningList.join(' | ')}</div>`;
                    } else {
                        runningOps.innerHTML = data.last_result ? 
                            `<div class="status status-idle">✅ ${data.last_result}</div>` : '';
                    }
                    setSummaryTile(
                        'sumOps',
                        'sumOpsNote',
                        String(runningList.length),
                        runningList.length ? runningList.join(' | ') : 'Idle',
                    );
                    
                    // Scan section
                    document.getElementById('stopScanBtn').style.display = data.scan_running ? 'inline-block' : 'none';
                    document.getElementById('scanBtn').disabled = data.scan_running;
                    const staleHint = (data.scan_stale_tickers == null) ? '' : ` · stale used: ${data.scan_stale_tickers}`;
                    document.getElementById('scanStatus').textContent = data.scan_running ? 'Running...' :
                        ((data.scan_last_result || '') + staleHint);
                    
                    const scheduleBtn = document.getElementById('scheduleBtn');
                    const cancelBtn = document.getElementById('cancelBtn');
                    const scheduleStatus = document.getElementById('scheduleStatus');
                    if (data.scheduled_time) {
                        scheduleBtn.style.display = 'none';
                        cancelBtn.style.display = 'inline-block';
                        let schedTxt = `📅 Next: ${data.scheduled_time} (${formatDays(data.scan_days)})`;
                        if (data.scan_interval_hours > 0) {
                            schedTxt += ` + every ${data.scan_interval_hours}h`;
                        }
                        scheduleStatus.innerHTML = `<p class="info" style="color:#f59e0b;">${schedTxt}</p>`;
                        setSummaryTile(
                            'sumNextScan',
                            'sumNextScanNote',
                            data.scheduled_time,
                            data.scan_interval_hours > 0
                                ? `${formatDays(data.scan_days)} · every ${data.scan_interval_hours}h`
                                : formatDays(data.scan_days),
                        );
                    } else {
                        scheduleBtn.style.display = 'inline-block';
                        cancelBtn.style.display = 'none';
                        scheduleStatus.innerHTML = '';
                        setSummaryTile('sumNextScan', 'sumNextScanNote', '-', 'Not scheduled');
                    }
                    
                    // Train section
                    document.getElementById('stopTrainBtn').style.display = data.train_running ? 'inline-block' : 'none';
                    document.getElementById('trainBtn').disabled = data.train_running;
                    document.getElementById('trainStatus').textContent = data.train_running ? 'Running...' : 
                        (data.train_last_result || '');
                    
                    const trainScheduleBtn = document.getElementById('trainScheduleBtn');
                    const trainCancelBtn = document.getElementById('trainCancelBtn');
                    const trainScheduleStatus = document.getElementById('trainScheduleStatus');
                    if (data.train_scheduled_time) {
                        trainScheduleBtn.style.display = 'none';
                        trainCancelBtn.style.display = 'inline-block';
                        trainScheduleStatus.innerHTML = `<p class="info" style="color:#7c3aed;">🧠 Next: ${data.train_scheduled_time}<br/>Every ${data.train_interval_days}d (${formatDays(data.train_days)})</p>`;
                    } else {
                        trainScheduleBtn.style.display = 'inline-block';
                        trainCancelBtn.style.display = 'none';
                        trainScheduleStatus.innerHTML = '';
                    }
                    
                    // Qual section
                    document.getElementById('stopQualBtn').style.display = data.qual_running ? 'inline-block' : 'none';
                    document.getElementById('qualBtn').disabled = data.qual_running;
                    document.getElementById('qualOpStatus').textContent = data.qual_running ? 'Running...' : 
                        (data.qual_last_result || '');
                    
                    const qualScheduleBtn = document.getElementById('qualScheduleBtn');
                    const qualCancelBtn = document.getElementById('qualCancelBtn');
                    const qualStatus = document.getElementById('qualStatus');
                    if (data.qual_scheduled_time) {
                        qualScheduleBtn.style.display = 'none';
                        qualCancelBtn.style.display = 'inline-block';
                        qualStatus.innerHTML = `<p class="info" style="color:#00d4ff;">🏭 Next: ${data.qual_scheduled_time}<br/>Every ${data.qual_interval_hours}h (${data.qual_chunk_size} tickers)</p>`;
                    } else {
                        qualScheduleBtn.style.display = 'inline-block';
                        qualCancelBtn.style.display = 'none';
                        qualStatus.innerHTML = '';
                    }

                    // News section
                    document.getElementById('stopNewsBtn').style.display = data.news_running ? 'inline-block' : 'none';
                    document.getElementById('newsBtn').disabled = data.news_running;
                    document.getElementById('newsOpStatus').textContent = data.news_running ? 'Running...' :
                        (data.news_last_result || '');

                    const newsScheduleBtn = document.getElementById('newsScheduleBtn');
                    const newsCancelBtn = document.getElementById('newsCancelBtn');
                    const newsStatus = document.getElementById('newsStatus');
                    if (data.news_scheduled_time) {
                        newsScheduleBtn.style.display = 'none';
                        newsCancelBtn.style.display = 'inline-block';
                        newsStatus.innerHTML = `<p class="info" style="color:#e879f9;">📰 Next: ${data.news_scheduled_time}<br/>Morning: ${data.news_morning_time} · Evening: ${data.news_evening_time} (${formatDays(data.news_days || [])})</p>`;
                    } else {
                        newsScheduleBtn.style.display = 'inline-block';
                        newsCancelBtn.style.display = 'none';
                        newsStatus.innerHTML = '';
                    }

                    // Audit section
                    document.getElementById('stopAuditBtn').style.display = data.audit_running ? 'inline-block' : 'none';
                    document.getElementById('auditBtn').disabled = data.audit_running;
                    document.getElementById('auditOpStatus').textContent = data.audit_running ? 'Running...' :
                        (data.audit_last_result || '');

                    const auditScheduleBtn = document.getElementById('auditScheduleBtn');
                    const auditCancelBtn = document.getElementById('auditCancelBtn');
                    const auditStatus = document.getElementById('auditStatus');
                    if (data.audit_scheduled_time) {
                        auditScheduleBtn.style.display = 'none';
                        auditCancelBtn.style.display = 'inline-block';
                        auditStatus.innerHTML = `<p class="info" style="color:#f59e0b;">📋 Next: ${data.audit_scheduled_time}<br/>Days: ${formatDays(data.audit_days || [])}</p>`;
                    } else {
                        auditScheduleBtn.style.display = 'inline-block';
                        auditCancelBtn.style.display = 'none';
                        auditStatus.innerHTML = '';
                    }

                    // Weekly summary section
                    document.getElementById('weeklySummaryBtn').disabled = data.weekly_summary_running;
                    document.getElementById('weeklySummaryOpStatus').textContent = data.weekly_summary_running ? 'Running...' :
                        (data.weekly_summary_last_result || '');
                    const weeklySummaryScheduleBtn = document.getElementById('weeklySummaryScheduleBtn');
                    const weeklySummaryCancelBtn = document.getElementById('weeklySummaryCancelBtn');
                    const weeklySummaryStatus = document.getElementById('weeklySummaryStatus');
                    if (data.weekly_summary_scheduled_time) {
                        weeklySummaryScheduleBtn.style.display = 'none';
                        weeklySummaryCancelBtn.style.display = 'inline-block';
                        weeklySummaryStatus.innerHTML = `<p class="info" style="color:#22c55e;">🗞️ Next: ${data.weekly_summary_scheduled_time}<br/>Day: ${DAYS[data.weekly_summary_day] || data.weekly_summary_day}</p>`;
                    } else {
                        weeklySummaryScheduleBtn.style.display = 'inline-block';
                        weeklySummaryCancelBtn.style.display = 'none';
                        weeklySummaryStatus.innerHTML = '';
                    }

                    // Snapshot section (monitor controls)
                    document.getElementById('stopSnapshotBtn').style.display = data.snapshot_running ? 'inline-block' : 'none';
                    document.getElementById('snapshotBtn').disabled = data.snapshot_running;
                    const snapshotScheduleBtn = document.getElementById('snapshotScheduleBtn');
                    const snapshotCancelBtn = document.getElementById('snapshotCancelBtn');
                    const snapshotScheduleStatus = document.getElementById('snapshotScheduleStatus');
                    if (data.snapshot_interval_minutes != null) {
                        document.getElementById('snapshotIntervalMinutes').value = data.snapshot_interval_minutes;
                    }
                    if (data.snapshot_recent_days != null) {
                        document.getElementById('snapshotRecentDays').value = data.snapshot_recent_days;
                    }
                    if (data.snapshot_scheduled_time) {
                        snapshotScheduleBtn.style.display = 'none';
                        snapshotCancelBtn.style.display = 'inline-block';
                        const lastRun = data.snapshot_last_run ? ` · last ${data.snapshot_last_run}` : '';
                        snapshotScheduleStatus.innerHTML = `<p class="info" style="color:#0ea5e9;">📸 Next: ${data.snapshot_scheduled_time} · every ${data.snapshot_interval_minutes}m${lastRun}</p>`;
                    } else {
                        snapshotScheduleBtn.style.display = 'inline-block';
                        snapshotCancelBtn.style.display = 'none';
                        const snapLast = data.snapshot_last_result ? `Last: ${data.snapshot_last_result}` : 'Auto snapshot off';
                        snapshotScheduleStatus.innerHTML = `<p class="info">${snapLast}</p>`;
                    }

                    // Sync server time
                    if (data.server_time) {
                        serverTimeOffset = new Date(data.server_time).getTime() - Date.now();
                    }
                });
        }
        setInterval(updateStatus, 3000);
        updateStatus();
        if (!configLoaded) loadConfigEditor();
        loadConfigHistory();
        loadPurgeSuggestions();
        setPreviewText('Select a report/log file', '');
        loadReportFiles();
        loadLogFiles();
        loadRunMonitor();
        setInterval(loadRunMonitor, 15000);
        
        function runScan() {
            fetch('/run/scan', {method: 'POST'}).then(updateStatus);
        }
        
        function runTrain() {
            fetch('/run/train', {method: 'POST'}).then(updateStatus);
        }
        
        function runQual() {
            const chunk = document.getElementById('qualChunk').value;
            fetch('/run/qual', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({chunk_size: parseInt(chunk)})
            }).then(updateStatus);
        }

        function runNews() {
            fetch('/run/news', {method: 'POST'}).then(updateStatus);
        }

        function setNewsSchedule() {
            const morningTime = document.getElementById('newsMorningTime').value;
            const eveningTime = document.getElementById('newsEveningTime').value;
            const days = getSelectedDays('newsDays');
            fetch('/news_schedule', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({morning_time: morningTime, evening_time: eveningTime, days: days})
            }).then(updateStatus);
        }

        function cancelNewsSchedule() {
            fetch('/news_schedule', {method: 'DELETE'}).then(updateStatus);
        }

        function runAudit() {
            fetch('/run/audit', {method: 'POST'}).then(updateStatus);
        }

        function runWeeklySummary() {
            fetch('/run/weekly_summary', {method: 'POST'}).then(updateStatus);
        }

        function runSnapshotNow() {
            const recentDays = parseInt(document.getElementById('snapshotRecentDays').value || '30');
            fetch('/run/snapshot', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({recent_days: recentDays}),
            }).then(() => {
                updateStatus();
                setTimeout(loadRunMonitor, 1200);
            });
        }
        
        function stopOp(op) {
            fetch('/stop/' + op, {method: 'POST'}).then(updateStatus);
        }
        
        function setSchedule() {
            const time = document.getElementById('scheduleTime').value;
            const days = getSelectedDays('scanDays');
            const repeatCheck = document.getElementById('scanRepeatCheck').checked;
            const intervalHours = repeatCheck ? parseInt(document.getElementById('scanIntervalHours').value) : 0;
            
            fetch('/schedule', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({time: time, days: days, interval_hours: intervalHours})
            }).then(updateStatus);
        }
        
        function cancelSchedule() {
            fetch('/schedule', {method: 'DELETE'}).then(updateStatus);
        }
        
        function setTrainSchedule() {
            const time = document.getElementById('trainTime').value;
            const days = document.getElementById('trainDays').value;
            const allowedDays = getSelectedDays('trainDayChecks');
            
            fetch('/train_schedule', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({time: time, days: parseInt(days), allowed_days: allowedDays})
            }).then(updateStatus);
        }
        
        function cancelTrainSchedule() {
            fetch('/train_schedule', {method: 'DELETE'}).then(updateStatus);
        }
        
        function setQualSchedule() {
            const chunk = document.getElementById('qualChunk').value;
            const interval = document.getElementById('qualInterval').value;
            fetch('/qual_schedule', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({chunk_size: parseInt(chunk), interval_hours: parseInt(interval)})
            }).then(updateStatus);
        }
        
        function cancelQualSchedule() {
            fetch('/qual_schedule', {method: 'DELETE'}).then(updateStatus);
        }

        function setSnapshotSchedule() {
            const interval = parseInt(document.getElementById('snapshotIntervalMinutes').value || '45');
            const recentDays = parseInt(document.getElementById('snapshotRecentDays').value || '30');
            fetch('/snapshot_schedule', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({interval_minutes: interval, recent_days: recentDays}),
            }).then(updateStatus);
        }

        function cancelSnapshotSchedule() {
            fetch('/snapshot_schedule', {method: 'DELETE'}).then(updateStatus);
        }

        function setAuditSchedule() {
            const time = document.getElementById('auditTime').value;
            const days = getSelectedDays('auditDays');
            fetch('/audit_schedule', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({time: time, days: days})
            }).then(updateStatus);
        }

        function cancelAuditSchedule() {
            fetch('/audit_schedule', {method: 'DELETE'}).then(updateStatus);
        }

        function setWeeklySummarySchedule() {
            const time = document.getElementById('weeklySummaryTime').value;
            const day = parseInt(document.getElementById('weeklySummaryDay').value);
            fetch('/weekly_summary_schedule', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({time: time, day: day})
            }).then(updateStatus);
        }

        function cancelWeeklySummarySchedule() {
            fetch('/weekly_summary_schedule', {method: 'DELETE'}).then(updateStatus);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, pid=os.getpid())


@app.route('/public/performance')
def public_performance():
    if not os.path.exists(PUBLIC_DASHBOARD_PATH):
        return jsonify({
            'error': 'Public dashboard not found yet',
            'hint': 'Run a scheduled scan or run: python src/main.py --pipeline daily_auto',
            'expected_path': PUBLIC_DASHBOARD_PATH,
        }), 404
    return send_file(PUBLIC_DASHBOARD_PATH)


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """Read/update selected configuration values from dashboard."""
    if request.method == 'GET':
        try:
            return jsonify({"status": "ok", "config": _config_snapshot()})
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    data = request.get_json(silent=True) or {}
    try:
        before_snapshot = _config_snapshot()
        updates = _dashboard_snapshot_to_config_updates(data)

        if not updates:
            return jsonify({"status": "noop", "message": "No updates provided.", "config": _config_snapshot()})

        changed = _update_config_file(updates)
        snapshot = _config_snapshot()
        history_id = int(time.time() * 1000)
        _append_config_history(
            {
                "id": history_id,
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "action": "update",
                "updated_keys": changed,
                "before": before_snapshot,
                "after": snapshot,
                "remote_addr": request.remote_addr or "",
            }
        )
        with STATE_LOCK:
            STATE["last_result"] = (
                f"Config updated @ {datetime.now().strftime('%Y-%m-%d %H:%M')} "
                f"({', '.join(changed)})"
            )
        return jsonify({"status": "ok", "updated_keys": changed, "history_id": history_id, "config": snapshot})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route('/api/config/history')
def api_config_history():
    """Return recent config changes for rollback/audit."""
    limit = _safe_int(request.args.get("limit"), 30) or 30
    rows = _read_config_history(limit=max(1, min(200, limit)))
    light_rows = []
    for row in rows:
        light_rows.append(
            {
                "id": row.get("id"),
                "ts": row.get("ts"),
                "action": row.get("action", "update"),
                "updated_keys": row.get("updated_keys", []),
                "remote_addr": row.get("remote_addr", ""),
            }
        )
    return jsonify({"status": "ok", "items": light_rows})


@app.route('/api/universe/purge', methods=['POST'])
def api_universe_purge():
    """Add one or more tickers to UNIVERSE_DENYLIST via a quick dashboard action."""
    data = request.get_json(silent=True) or {}
    raw_tickers = data.get("tickers", [])
    to_purge = _normalize_ticker_list(raw_tickers)
    if not to_purge:
        return jsonify({"status": "error", "message": "No tickers provided."}), 400

    try:
        before_snapshot = _config_snapshot()
        existing = _normalize_ticker_list(before_snapshot.get("universe_denylist", []))
        existing_set = set(existing)

        merged = list(existing)
        added = []
        for ticker in to_purge:
            if ticker in existing_set:
                continue
            merged.append(ticker)
            existing_set.add(ticker)
            added.append(ticker)

        if not added:
            return jsonify(
                {
                    "status": "noop",
                    "message": "Tickers already in denylist.",
                    "added": [],
                    "total_denylist": len(existing),
                    "config": before_snapshot,
                }
            )

        changed = _update_config_file({"UNIVERSE_DENYLIST": merged})
        snapshot = _config_snapshot()
        history_id = int(time.time() * 1000)
        _append_config_history(
            {
                "id": history_id,
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "action": "purge_universe",
                "updated_keys": changed,
                "added_tickers": added,
                "before": before_snapshot,
                "after": snapshot,
                "remote_addr": request.remote_addr or "",
            }
        )
        with STATE_LOCK:
            STATE["last_result"] = (
                f"Universe purge @ {datetime.now().strftime('%Y-%m-%d %H:%M')} "
                f"(added {len(added)})"
            )
        return jsonify(
            {
                "status": "ok",
                "added": added,
                "total_denylist": len(snapshot.get("universe_denylist", []) or []),
                "history_id": history_id,
                "config": snapshot,
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route('/api/universe/purge_suggestions')
@app.route('/api/universe/purge-suggestions')
def api_universe_purge_suggestions():
    """
    Suggest tickers to purge from the scan universe based on weak liquidity,
    stale data, and low model activity.
    """
    limit = _safe_int(request.args.get("limit"), 25) or 25
    limit = max(1, min(200, limit))
    if not os.path.exists(DB_PATH):
        return jsonify({"status": "error", "message": "Market DB not found."}), 404

    try:
        cfg = _load_config_module()
        portfolio = set(_normalize_ticker_list(getattr(cfg, "PORTFOLIO_HOLDINGS", []) or []))
        watchlist = set(_normalize_ticker_list(getattr(cfg, "WATCHLIST", []) or []))
        denylist = set(_normalize_ticker_list(getattr(cfg, "UNIVERSE_DENYLIST", []) or []))
        safe_assets = set(_normalize_ticker_list(getattr(cfg, "SAFE_ASSET_ALLOWLIST", []) or []))
        protected = portfolio | watchlist | safe_assets | denylist

        vol_floor_cfg = float(getattr(cfg, "TRADABILITY_MIN_AVG_VOLUME_20D", 300000) or 300000)
        dollar_floor_cfg = float(getattr(cfg, "TRADABILITY_MIN_AVG_DOLLAR_VOLUME_20D", 5000000) or 5000000)
        price_floor_cfg = float(getattr(cfg, "TRADABILITY_MIN_PRICE", 3.0) or 3.0)

        volume_floor = max(50000.0, vol_floor_cfg * 0.25)
        dollar_floor = max(750000.0, dollar_floor_cfg * 0.25)
        price_floor = max(2.0, price_floor_cfg * 0.67)
        stale_days_cutoff = 3

        con = duckdb.connect(DB_PATH, read_only=True)
        query = """
        WITH ranked AS (
            SELECT
                UPPER(ticker) AS ticker,
                date,
                close,
                volume,
                ROW_NUMBER() OVER (PARTITION BY UPPER(ticker) ORDER BY date DESC) AS rn
            FROM price_history
        ),
        last_bar AS (
            SELECT ticker, MAX(date) AS last_date
            FROM ranked
            GROUP BY ticker
        ),
        liq20 AS (
            SELECT
                ticker,
                AVG(close) AS avg_close_20d,
                AVG(volume) AS avg_volume_20d,
                AVG(close * volume) AS avg_dollar_volume_20d,
                COUNT(*) AS bars_20d
            FROM ranked
            WHERE rn <= 20
            GROUP BY ticker
        ),
        pred60 AS (
            SELECT
                UPPER(ticker) AS ticker,
                COUNT(*) AS pred_rows_60d
            FROM model_predictions
            WHERE asof_date >= current_date - 60
            GROUP BY UPPER(ticker)
        )
        SELECT
            b.ticker,
            b.last_date,
            q.avg_close_20d,
            q.avg_volume_20d,
            q.avg_dollar_volume_20d,
            COALESCE(p.pred_rows_60d, 0) AS pred_rows_60d
        FROM last_bar b
        JOIN liq20 q ON q.ticker = b.ticker
        LEFT JOIN pred60 p ON p.ticker = b.ticker
        """
        rows = con.execute(query).fetchall()
        con.close()

        today = datetime.now().date()
        suggestions = []
        for ticker, last_date, avg_close, avg_vol, avg_dollar, pred_rows in rows:
            symbol = str(ticker or "").upper().strip()
            if not symbol or symbol in protected:
                continue

            reasons = []
            score = 0

            try:
                stale_days = max((today - last_date).days, 0) if last_date else 999
            except Exception:
                stale_days = 999
            if stale_days > stale_days_cutoff:
                reasons.append(f"stale_data>{stale_days_cutoff}d")
                score += 3

            avg_close_val = float(avg_close or 0.0)
            avg_vol_val = float(avg_vol or 0.0)
            avg_dollar_val = float(avg_dollar or 0.0)
            pred_rows_val = int(pred_rows or 0)

            if avg_close_val < price_floor:
                reasons.append("low_price")
                score += 2
            if avg_vol_val < volume_floor:
                reasons.append("low_volume")
                score += 2
            if avg_dollar_val < dollar_floor:
                reasons.append("low_dollar_volume")
                score += 3
            if pred_rows_val == 0:
                reasons.append("no_recent_predictions")
                score += 1

            if score < 3:
                continue

            suggestions.append(
                {
                    "ticker": symbol,
                    "score": int(score),
                    "reasons": reasons,
                    "last_date": last_date.strftime("%Y-%m-%d") if last_date else "",
                    "stale_days": int(stale_days),
                    "avg_close_20d": avg_close_val,
                    "avg_volume_20d": avg_vol_val,
                    "avg_dollar_volume_20d": avg_dollar_val,
                    "pred_rows_60d": pred_rows_val,
                }
            )

        suggestions.sort(
            key=lambda x: (
                -int(x.get("score", 0)),
                float(x.get("avg_dollar_volume_20d", 0.0)),
                float(x.get("avg_volume_20d", 0.0)),
                x.get("ticker", ""),
            )
        )
        return jsonify(
            {
                "status": "ok",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "thresholds": {
                    "price_floor": price_floor,
                    "volume_floor": volume_floor,
                    "dollar_volume_floor": dollar_floor,
                    "stale_days_cutoff": stale_days_cutoff,
                },
                "excluded_counts": {
                    "portfolio": len(portfolio),
                    "watchlist": len(watchlist),
                    "safe_assets": len(safe_assets),
                    "denylist": len(denylist),
                },
                "total_candidates": len(suggestions),
                "suggestions": suggestions[:limit],
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route('/api/config/rollback', methods=['POST'])
def api_config_rollback():
    """Rollback config to the 'before' snapshot of a history entry."""
    data = request.get_json(silent=True) or {}
    target_id = str(data.get("id", "")).strip()
    if not target_id:
        return jsonify({"status": "error", "message": "Missing history id"}), 400
    rows = _read_config_history(limit=500)
    target = None
    for row in rows:
        if str(row.get("id", "")) == target_id:
            target = row
            break
    if target is None:
        return jsonify({"status": "error", "message": "History id not found"}), 404
    before = target.get("before")
    if not isinstance(before, dict):
        return jsonify({"status": "error", "message": "Selected history entry has no rollback snapshot"}), 400
    try:
        pre_snapshot = _config_snapshot()
        updates = _dashboard_snapshot_to_config_updates(before)
        changed = _update_config_file(updates)
        post_snapshot = _config_snapshot()
        history_id = int(time.time() * 1000)
        _append_config_history(
            {
                "id": history_id,
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "action": "rollback",
                "updated_keys": changed,
                "rollback_from_id": target_id,
                "before": pre_snapshot,
                "after": post_snapshot,
                "remote_addr": request.remote_addr or "",
            }
        )
        with STATE_LOCK:
            STATE["last_result"] = (
                f"Config rollback @ {datetime.now().strftime('%Y-%m-%d %H:%M')} "
                f"(from {target_id})"
            )
        return jsonify({"status": "ok", "updated_keys": changed, "history_id": history_id, "config": post_snapshot})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route('/api/run_monitor')
def api_run_monitor():
    """Live monitor payload for long pipeline/replay jobs."""
    try:
        return jsonify({"status": "ok", "monitor": _collect_run_monitor()})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route('/api/reports')
def api_reports():
    """List recent report files for UI browsing."""
    files = _list_files(
        REPORTS_DIR,
        exts=(".md", ".json", ".html", ".csv", ".txt"),
        max_files=120,
    )
    return jsonify({"status": "ok", "base": REPORTS_DIR, "files": files})


@app.route('/api/logs')
def api_logs():
    """List recent log files for UI browsing."""
    files = _list_files(
        LOGS_DIR,
        exts=(".log", ".txt", ".json", ".md"),
        max_files=120,
    )
    return jsonify({"status": "ok", "base": LOGS_DIR, "files": files})


@app.route('/api/file_preview')
def api_file_preview():
    """Return safe text preview for reports/logs/public files."""
    kind = (request.args.get("kind") or "reports").strip().lower()
    rel_path = request.args.get("path") or ""
    max_chars = _safe_int(request.args.get("max_chars"), 30000) or 30000
    try:
        payload = _file_preview(kind=kind, rel_path=rel_path, max_chars=max_chars)
        return jsonify({"status": "ok", **payload})
    except FileNotFoundError:
        return jsonify({"status": "error", "message": "File not found"}), 404
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route('/files/open')
def open_file():
    """Serve report/log files by relative path."""
    kind = (request.args.get("kind") or "reports").strip().lower()
    rel_path = request.args.get("path") or ""
    try:
        abs_path = _safe_join(_resolve_kind_base_dir(kind), rel_path)
        return send_file(abs_path)
    except FileNotFoundError:
        return jsonify({"status": "error", "message": "File not found"}), 404
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route('/status')
def status():
    # Check for external PID file (synced from remote)
    pid_file = os.path.join(PROJECT_ROOT, 'market_scan.pid')
    external_running = os.path.exists(pid_file)
    
    with STATE_LOCK:
        # Check if scan is also running externally
        scan_running = STATE.get('scan_running', False) or external_running
        
        return jsonify({
            'server_time': datetime.now().isoformat(),
            # Per-operation running flags
            'scan_running': scan_running,
            'train_running': STATE.get('train_running', False),
            'qual_running': STATE.get('qual_running', False),
            'news_running': STATE.get('news_running', False),
            'audit_running': STATE.get('audit_running', False),
            'weekly_summary_running': STATE.get('weekly_summary_running', False),
            'snapshot_running': STATE.get('snapshot_running', False),
            # Scan
            'scheduled_time': STATE['scheduled_time'].strftime('%Y-%m-%d %H:%M') if STATE.get('scheduled_time') else None,
            'scan_days': STATE.get('scan_days', []),
            'scan_interval_hours': STATE.get('scan_interval_hours', 0),
            'scan_last_result': STATE.get('scan_last_result'),
            'scan_stale_tickers': STATE.get('scan_stale_tickers'),
            'scan_failed_fetch_tickers': STATE.get('scan_failed_fetch_tickers'),
            'scan_insufficient_history_tickers': STATE.get('scan_insufficient_history_tickers'),
            # Train
            'train_scheduled_time': STATE['train_scheduled_time'].strftime('%Y-%m-%d %H:%M') if STATE.get('train_scheduled_time') else None,
            'train_interval_days': STATE.get('train_interval_days', 10),
            'train_days': STATE.get('train_days', []),
            'train_last_result': STATE.get('train_last_result'),
            # Qual
            'qual_scheduled_time': STATE['qual_scheduled_time'].strftime('%Y-%m-%d %H:%M') if STATE.get('qual_scheduled_time') else None,
            'qual_chunk_size': STATE.get('qual_chunk_size', 20),
            'qual_interval_hours': STATE.get('qual_interval_hours', 4),
            'qual_last_result': STATE.get('qual_last_result'),
            # News
            'news_scheduled_time': STATE['news_scheduled_time'].strftime('%Y-%m-%d %H:%M') if STATE.get('news_scheduled_time') else None,
            'news_morning_time': STATE.get('news_morning_time', '08:30'),
            'news_evening_time': STATE.get('news_evening_time', '16:30'),
            'news_days': STATE.get('news_days', [0, 1, 2, 3, 4]),
            'news_last_result': STATE.get('news_last_result'),
            # Audit
            'audit_scheduled_time': STATE['audit_scheduled_time'].strftime('%Y-%m-%d %H:%M') if STATE.get('audit_scheduled_time') else None,
            'audit_days': STATE.get('audit_days', [0, 1, 2, 3, 4]),
            'audit_time': STATE.get('audit_time', DASHBOARD_AUDIT_DEFAULT_TIME),
            'audit_last_result': STATE.get('audit_last_result'),
            # Weekly summary
            'weekly_summary_scheduled_time': STATE['weekly_summary_scheduled_time'].strftime('%Y-%m-%d %H:%M') if STATE.get('weekly_summary_scheduled_time') else None,
            'weekly_summary_day': STATE.get('weekly_summary_day', 4),
            'weekly_summary_time': STATE.get('weekly_summary_time', DASHBOARD_WEEKLY_SUMMARY_DEFAULT_TIME),
            'weekly_summary_last_result': STATE.get('weekly_summary_last_result'),
            # Snapshot
            'snapshot_scheduled_time': STATE['snapshot_scheduled_time'].strftime('%Y-%m-%d %H:%M') if STATE.get('snapshot_scheduled_time') else None,
            'snapshot_interval_minutes': STATE.get('snapshot_interval_minutes', 45),
            'snapshot_recent_days': STATE.get('snapshot_recent_days', 30),
            'snapshot_last_run': STATE.get('snapshot_last_run'),
            'snapshot_last_result': STATE.get('snapshot_last_result'),
            # Legacy
            'last_result': STATE.get('last_result'),
        })

@app.route('/stop/<operation>', methods=['POST'])
def stop_operation(operation):
    """Stop a specific operation (scan, train, qual, news, audit, weekly_summary, snapshot)."""
    with PROCESS_LOCK:
        proc = PROCESSES.get(operation)
        if proc:
            try:
                if sys.platform != 'win32':
                    import signal
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
                
                PROCESSES[operation] = None
                
                with STATE_LOCK:
                    STATE[f'{operation}_running'] = False
                    result_msg = f"Stopped {operation} @ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    STATE[f'{operation}_last_result'] = result_msg
                    STATE['last_result'] = result_msg
                
                return jsonify({'status': 'stopped', 'operation': operation})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    # For scan, also check external PID file
    if operation == 'scan':
        pid_file = os.path.join(PROJECT_ROOT, 'market_scan.pid')
        if os.path.exists(pid_file):
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                import signal
                os.kill(pid, signal.SIGTERM)
                
                try:
                    os.remove(pid_file)
                except:
                    pass
                
                with STATE_LOCK:
                    result_msg = f"Stopped external scan (PID {pid}) @ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    STATE['scan_last_result'] = result_msg
                    STATE['last_result'] = result_msg
                
                return jsonify({'status': 'stopped', 'message': f'External process {pid} killed'})
            except ProcessLookupError:
                try:
                    os.remove(pid_file)
                except:
                    pass
                return jsonify({'status': 'stopped', 'message': 'Process was already dead'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    return jsonify({'status': 'not_running', 'operation': operation})


@app.route('/run/scan', methods=['POST'])
def run_scan():
    """Run scan operation."""
    with STATE_LOCK:
        if STATE.get('scan_running'):
            return jsonify({'error': 'Scan already running'}), 400
        STATE['scan_running'] = True
        STATE['scan_stale_tickers'] = None
        STATE['scan_failed_fetch_tickers'] = None
        STATE['scan_insufficient_history_tickers'] = None
    
    threading.Thread(target=lambda: run_op_internal('scan', scheduled=False), daemon=True).start()
    return jsonify({'status': 'started', 'operation': 'scan'})


@app.route('/run/train', methods=['POST'])
def run_train():
    """Run train operation."""
    with STATE_LOCK:
        if STATE.get('train_running'):
            return jsonify({'error': 'Training already running'}), 400
        STATE['train_running'] = True
    
    threading.Thread(target=lambda: run_op_internal('train'), daemon=True).start()
    return jsonify({'status': 'started', 'operation': 'train'})


@app.route('/run/news', methods=['POST'])
def run_news():
    """Run news assessment now (fetches via Gemini CLI)."""
    with STATE_LOCK:
        if STATE.get('news_running'):
            return jsonify({'error': 'News already running'}), 400
        STATE['news_running'] = True

    threading.Thread(target=lambda: run_op_internal('news'), daemon=True).start()
    return jsonify({'status': 'started', 'operation': 'news'})


@app.route('/run/audit', methods=['POST'])
def run_audit():
    """Run audit operation."""
    with STATE_LOCK:
        if STATE.get('audit_running'):
            return jsonify({'error': 'Audit already running'}), 400
        STATE['audit_running'] = True

    threading.Thread(target=lambda: run_op_internal('audit'), daemon=True).start()
    return jsonify({'status': 'started', 'operation': 'audit'})


@app.route('/run/weekly_summary', methods=['POST'])
def run_weekly_summary():
    """Send weekly summary notification now."""
    with STATE_LOCK:
        if STATE.get('weekly_summary_running'):
            return jsonify({'error': 'Weekly summary already running'}), 400
        STATE['weekly_summary_running'] = True

    threading.Thread(target=lambda: run_op_internal('weekly_summary'), daemon=True).start()
    return jsonify({'status': 'started', 'operation': 'weekly_summary'})


@app.route('/run/snapshot', methods=['POST'])
def run_snapshot():
    """Generate analytics snapshot now (read-only DB/report extraction)."""
    data = request.get_json(silent=True) or {}
    recent_days = _safe_int(data.get('recent_days'), STATE.get('snapshot_recent_days', 30)) or 30
    with STATE_LOCK:
        if STATE.get('snapshot_running'):
            return jsonify({'error': 'Snapshot already running'}), 400
        STATE['snapshot_running'] = True
        STATE['snapshot_recent_days'] = max(1, min(365, recent_days))

    threading.Thread(target=lambda: run_op_internal('snapshot'), daemon=True).start()
    return jsonify({'status': 'started', 'operation': 'snapshot', 'recent_days': STATE.get('snapshot_recent_days', 30)})



def get_next_valid_time(target_dt, days_interval=1, allowed_days=None):
    """
    Advance target_dt by days_interval, then ensure it lands on an allowed_day.
    allowed_days: list of ints [0..6] (Mon..Sun). If None/empty, allow all.
    """
    next_run = target_dt + timedelta(days=days_interval)
    
    if not allowed_days:
        return next_run
        
    # Safety: avoid infinite loop if allowed_days is somehow invalid (e.g. empty)
    # But usually front end sends at least one day. Defaults handle None.
    
    # Advance until weekday is in allowed_days
    # Max loop 14 just to be safe (2 weeks)
    for _ in range(14):
        if next_run.weekday() in allowed_days:
            return next_run
        next_run += timedelta(days=1)
    
    return next_run # Fallback if loop exhausted


def _persist_scan_schedule_locked():
    """Persist scan schedule settings so restarts do not drop the schedule."""
    payload = {
        "scheduled_time": (
            STATE["scheduled_time"].strftime("%Y-%m-%d %H:%M:%S")
            if STATE.get("scheduled_time")
            else None
        ),
        "scan_days": _sanitize_days(STATE.get("scan_days", [0, 1, 2, 3, 4]), fallback=[0, 1, 2, 3, 4]),
        "scan_interval_hours": max(0, int(STATE.get("scan_interval_hours", 0) or 0)),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(SCAN_SCHEDULE_STATE_PATH), exist_ok=True)
    tmp_path = f"{SCAN_SCHEDULE_STATE_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp_path, SCAN_SCHEDULE_STATE_PATH)


def _load_scan_schedule_state():
    """Restore persisted scan schedule settings on startup."""
    if not os.path.exists(SCAN_SCHEDULE_STATE_PATH):
        return

    try:
        with open(SCAN_SCHEDULE_STATE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return

    target = None
    raw_target = raw.get("scheduled_time")
    if isinstance(raw_target, str) and raw_target.strip():
        for dt_fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                target = datetime.strptime(raw_target.strip(), dt_fmt)
                break
            except ValueError:
                continue

    days = _sanitize_days(raw.get("scan_days", [0, 1, 2, 3, 4]), fallback=[0, 1, 2, 3, 4])
    try:
        interval_hours = max(0, int(raw.get("scan_interval_hours", 0) or 0))
    except Exception:
        interval_hours = 0

    with STATE_LOCK:
        STATE["scheduled_time"] = target
        STATE["scan_days"] = days
        STATE["scan_interval_hours"] = interval_hours


def _start_scan_scheduler_once():
    """Start a single background scheduler loop for scans."""
    global SCAN_SCHEDULER_STARTED
    if SCAN_SCHEDULER_STARTED:
        return
    SCAN_SCHEDULER_STARTED = True

    def scan_scheduler_loop():
        while True:
            should_trigger = False
            with STATE_LOCK:
                target = STATE.get("scheduled_time")
                if target is not None:
                    now = datetime.now()
                    window = STATE.get("schedule_window_minutes", 5)
                    allow = _sanitize_days(STATE.get("scan_days", [0, 1, 2, 3, 4]), fallback=[0, 1, 2, 3, 4])
                    interval = max(0, int(STATE.get("scan_interval_hours", 0) or 0))

                    diff = (now - target).total_seconds() / 60
                    if now >= target:
                        if abs(diff) <= window:
                            if not STATE.get("scan_running"):
                                STATE["scan_running"] = True
                                if interval > 0:
                                    next_run = now + timedelta(hours=interval)
                                else:
                                    next_run = get_next_valid_time(target, 1, allow)
                                STATE["scheduled_time"] = next_run
                                STATE["last_result"] = f"Scan triggered @ {now.strftime('%Y-%m-%d %H:%M')}"
                                _persist_scan_schedule_locked()
                                should_trigger = True
                        elif diff > window:
                            if interval > 0:
                                next_run = now + timedelta(hours=interval)
                            else:
                                next_run = get_next_valid_time(target, 1, allow)
                            STATE["scheduled_time"] = next_run
                            STATE["last_result"] = f"Missed scan, next: {next_run.strftime('%Y-%m-%d %H:%M')}"
                            _persist_scan_schedule_locked()

            if should_trigger:
                threading.Thread(target=lambda: run_op_internal("scan", scheduled=True), daemon=True).start()
            time.sleep(30)

    threading.Thread(target=scan_scheduler_loop, daemon=True).start()


def _start_news_scheduler_once():
    """Start a single background scheduler loop for news assessments (morning + evening)."""
    global NEWS_SCHEDULER_STARTED
    if NEWS_SCHEDULER_STARTED:
        return
    NEWS_SCHEDULER_STARTED = True

    def news_scheduler_loop():
        while True:
            with STATE_LOCK:
                target = STATE.get('news_scheduled_time')
                if target is not None:
                    now = datetime.now()
                    window = STATE.get('schedule_window_minutes', 5)
                    allow = _sanitize_days(STATE.get('news_days', [0, 1, 2, 3, 4]), fallback=[0, 1, 2, 3, 4])
                    morning_h, morning_m = _parse_hour_minute(STATE.get('news_morning_time', '08:30'), '08:30')
                    evening_h, evening_m = _parse_hour_minute(STATE.get('news_evening_time', '16:30'), '16:30')
                    diff = (now - target).total_seconds() / 60

                    if now >= target:
                        if abs(diff) <= window:
                            if not STATE.get('news_running'):
                                STATE['news_running'] = True
                                # Determine next run: if this was morning, schedule evening today;
                                # if this was evening, schedule next morning on allowed day
                                cur_h = target.hour
                                cur_m = target.minute
                                if cur_h == morning_h and cur_m == morning_m:
                                    # Schedule evening run today
                                    next_run = now.replace(hour=evening_h, minute=evening_m, second=0, microsecond=0)
                                    if next_run <= now:
                                        next_run = _next_run_for_days(morning_h, morning_m, allow, now=now + timedelta(minutes=1))
                                else:
                                    # Schedule morning run on next allowed day
                                    next_run = _next_run_for_days(morning_h, morning_m, allow, now=now + timedelta(minutes=1))
                                STATE['news_scheduled_time'] = next_run
                                STATE['last_result'] = f"News triggered @ {now.strftime('%Y-%m-%d %H:%M')}"
                                threading.Thread(target=lambda: run_op_internal('news', scheduled=True), daemon=True).start()
                        elif diff > window:
                            # Missed window: schedule next morning on allowed day
                            next_run = _next_run_for_days(morning_h, morning_m, allow, now=now + timedelta(minutes=1))
                            STATE['news_scheduled_time'] = next_run
                            STATE['last_result'] = f"Missed news, next: {next_run.strftime('%Y-%m-%d %H:%M')}"

            time.sleep(30)

    threading.Thread(target=news_scheduler_loop, daemon=True).start()


def _start_audit_scheduler_once():
    """Start a single background scheduler loop for audits."""
    global AUDIT_SCHEDULER_STARTED
    if AUDIT_SCHEDULER_STARTED:
        return
    AUDIT_SCHEDULER_STARTED = True

    def audit_scheduler_loop():
        while True:
            with STATE_LOCK:
                target = STATE.get('audit_scheduled_time')
                if target is None:
                    pass
                else:
                    now = datetime.now()
                    allow = _sanitize_days(STATE.get('audit_days', [0, 1, 2, 3, 4]), fallback=[0, 1, 2, 3, 4])
                    hour, minute = _parse_hour_minute(STATE.get('audit_time', DASHBOARD_AUDIT_DEFAULT_TIME), DASHBOARD_AUDIT_DEFAULT_TIME)
                    window = STATE.get('schedule_window_minutes', 5)
                    diff = (now - target).total_seconds() / 60

                    if now >= target:
                        if abs(diff) <= window:
                            if not STATE.get('audit_running'):
                                STATE['audit_running'] = True
                                STATE['audit_scheduled_time'] = _next_run_for_days(hour, minute, allow, now=now + timedelta(minutes=1))
                                STATE['last_result'] = f"Audit triggered @ {now.strftime('%Y-%m-%d %H:%M')}"
                                threading.Thread(target=lambda: run_op_internal('audit', scheduled=True), daemon=True).start()
                        elif diff > window:
                            next_run = _next_run_for_days(hour, minute, allow, now=now + timedelta(minutes=1))
                            STATE['audit_scheduled_time'] = next_run
                            STATE['last_result'] = f"Missed audit, next: {next_run.strftime('%Y-%m-%d %H:%M')}"
            time.sleep(30)

    threading.Thread(target=audit_scheduler_loop, daemon=True).start()


def _start_weekly_summary_scheduler_once():
    """Start a single background scheduler loop for weekly summaries."""
    global WEEKLY_SUMMARY_SCHEDULER_STARTED
    if WEEKLY_SUMMARY_SCHEDULER_STARTED:
        return
    WEEKLY_SUMMARY_SCHEDULER_STARTED = True

    def weekly_summary_scheduler_loop():
        while True:
            with STATE_LOCK:
                target = STATE.get('weekly_summary_scheduled_time')
                if target is None:
                    pass
                else:
                    now = datetime.now()
                    day = _sanitize_days([STATE.get('weekly_summary_day', 4)], fallback=[4])[0]
                    hour, minute = _parse_hour_minute(
                        STATE.get('weekly_summary_time', DASHBOARD_WEEKLY_SUMMARY_DEFAULT_TIME),
                        DASHBOARD_WEEKLY_SUMMARY_DEFAULT_TIME,
                    )
                    window = STATE.get('schedule_window_minutes', 5)
                    diff = (now - target).total_seconds() / 60

                    if now >= target:
                        if abs(diff) <= window:
                            if not STATE.get('weekly_summary_running'):
                                STATE['weekly_summary_running'] = True
                                STATE['weekly_summary_scheduled_time'] = _next_run_for_days(
                                    hour, minute, [day], now=now + timedelta(minutes=1)
                                )
                                STATE['last_result'] = f"Weekly summary triggered @ {now.strftime('%Y-%m-%d %H:%M')}"
                                threading.Thread(target=lambda: run_op_internal('weekly_summary'), daemon=True).start()
                        elif diff > window:
                            next_run = _next_run_for_days(hour, minute, [day], now=now + timedelta(minutes=1))
                            STATE['weekly_summary_scheduled_time'] = next_run
                            STATE['last_result'] = f"Missed weekly summary, next: {next_run.strftime('%Y-%m-%d %H:%M')}"
            time.sleep(30)

    threading.Thread(target=weekly_summary_scheduler_loop, daemon=True).start()


def _start_snapshot_scheduler_once():
    """Start a single background scheduler loop for analytics snapshots."""
    global SNAPSHOT_SCHEDULER_STARTED
    if SNAPSHOT_SCHEDULER_STARTED:
        return
    SNAPSHOT_SCHEDULER_STARTED = True

    def snapshot_scheduler_loop():
        while True:
            should_trigger = False
            with STATE_LOCK:
                target = STATE.get('snapshot_scheduled_time')
                interval = max(5, int(STATE.get('snapshot_interval_minutes', 45) or 45))
                window = STATE.get('schedule_window_minutes', 5)
                now = datetime.now()

                if target is not None:
                    diff = (now - target).total_seconds() / 60
                    if now >= target:
                        if abs(diff) <= window:
                            if not STATE.get('snapshot_running'):
                                STATE['snapshot_running'] = True
                                STATE['snapshot_scheduled_time'] = now + timedelta(minutes=interval)
                                STATE['last_result'] = f"Snapshot triggered @ {now.strftime('%Y-%m-%d %H:%M')}"
                                should_trigger = True
                        elif diff > window:
                            next_run = now + timedelta(minutes=interval)
                            STATE['snapshot_scheduled_time'] = next_run
                            STATE['last_result'] = f"Missed snapshot, next: {next_run.strftime('%Y-%m-%d %H:%M')}"

            if should_trigger:
                threading.Thread(target=lambda: run_op_internal('snapshot'), daemon=True).start()
            time.sleep(20)

    threading.Thread(target=snapshot_scheduler_loop, daemon=True).start()


@app.route('/schedule', methods=['POST', 'DELETE'])
def schedule():
    if request.method == 'DELETE':
        with STATE_LOCK:
            STATE['scheduled_time'] = None
            _persist_scan_schedule_locked()
        return jsonify({'status': 'cancelled'})
    
    data = request.get_json(silent=True) or {}
    time_str = data.get('time', '09:30')
    allowed_days = _sanitize_days(data.get('days', [0, 1, 2, 3, 4]), fallback=[0, 1, 2, 3, 4])
    try:
        interval_hours = max(0, int(data.get('interval_hours', 0) or 0))  # 0 = daily only
    except Exception:
        interval_hours = 0
    
    hour, minute = _parse_hour_minute(time_str, default='09:30')
    
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    if target <= now:
        target += timedelta(days=1)
    
    while target.weekday() not in allowed_days:
        target += timedelta(days=1)
    
    with STATE_LOCK:
        STATE['scheduled_time'] = target
        STATE['scan_days'] = allowed_days
        STATE['scan_interval_hours'] = interval_hours
        _persist_scan_schedule_locked()

    _start_scan_scheduler_once()
    return jsonify({'status': 'scheduled', 'time': target.strftime('%Y-%m-%d %H:%M')})


def run_op_internal(operation, scheduled=False):
    """Run an operation with per-operation process tracking."""
    # Defensive check: ensure we don't run concurrent operations of the same type.
    # The scheduler sets *_running = True before spawning this thread, but if the
    # thread completes very quickly (e.g. immediate failure) there's a narrow window
    # where another invocation could slip through.
    with STATE_LOCK:
        if STATE.get(f'{operation}_running') and PROCESSES.get(operation) is not None:
            return  # Already running — skip this invocation
        STATE[f'{operation}_running'] = True
    try:
        if operation == 'scan':
            if scheduled:
                cmd = [sys.executable, 'src/main.py', '--pipeline', 'daily_auto']
            else:
                cmd = [sys.executable, 'src/main.py', '--scan']
        elif operation == 'train':
            cmd = [sys.executable, 'src/main.py', '--train-ml']
        elif operation == 'news':
            cmd = [sys.executable, 'src/main.py', '--update-news']
        elif operation == 'qual':
            chunk = STATE.get('qual_chunk_size', 20)
            cmd = [sys.executable, 'scripts/update_qual_features.py', '--count', str(chunk)]
        elif operation == 'audit':
            cmd = [sys.executable, 'src/main.py', '--pipeline', 'audit,notify']
        elif operation == 'weekly_summary':
            cmd = [sys.executable, 'src/main.py', '--notify-weekly']
        elif operation == 'snapshot':
            recent_days = max(1, int(STATE.get('snapshot_recent_days', 30) or 30))
            cmd = [sys.executable, 'scripts/generate_analytics_snapshot.py', '--recent-days', str(recent_days)]
        else:
            return
        
        with PROCESS_LOCK:
            if sys.platform != 'win32':
                proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       preexec_fn=os.setsid)
            else:
                proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            PROCESSES[operation] = proc
        
        stdout, stderr = proc.communicate()
        returncode = proc.returncode
        stdout_text = stdout.decode('utf-8', errors='ignore') if isinstance(stdout, (bytes, bytearray)) else str(stdout or '')
        stderr_text = stderr.decode('utf-8', errors='ignore') if isinstance(stderr, (bytes, bytearray)) else str(stderr or '')
        combined_text = f"{stdout_text}\n{stderr_text}".strip()
        scan_meta_match = None
        if operation == 'scan':
            scan_meta_match = re.search(
                r"SCAN_META\s+total_tickers=(\d+)\s+results_generated=(\d+)\s+"
                r"stale_tickers=(\d+)\s+failed_fetch_tickers=(\d+)\s+insufficient_history_tickers=(\d+)",
                combined_text,
            )
        
        with STATE_LOCK:
            STATE[f'{operation}_running'] = False
            if operation == 'snapshot':
                STATE['snapshot_last_run'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if operation == 'scan':
                if scan_meta_match:
                    STATE['scan_stale_tickers'] = int(scan_meta_match.group(3))
                    STATE['scan_failed_fetch_tickers'] = int(scan_meta_match.group(4))
                    STATE['scan_insufficient_history_tickers'] = int(scan_meta_match.group(5))
                else:
                    STATE['scan_stale_tickers'] = None
                    STATE['scan_failed_fetch_tickers'] = None
                    STATE['scan_insufficient_history_tickers'] = None
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            op_label = operation.replace('_', ' ').title()
            if returncode == 0:
                result_msg = f"{op_label} OK @ {timestamp}"
                if operation == 'scan' and scan_meta_match:
                    result_msg = (
                        f"{op_label} OK @ {timestamp} "
                        f"(stale={int(scan_meta_match.group(3))}, "
                        f"fetch_failures={int(scan_meta_match.group(4))})"
                    )
                if operation == 'qual':
                    if 'All tickers up to date!' in combined_text:
                        result_msg = f"{op_label} OK @ {timestamp} (no stale tickers)"
                    else:
                        m = re.search(r'Complete:\s*(\d+)\s*success,\s*(\d+)\s*failed', combined_text)
                        if m:
                            result_msg = f"{op_label} OK @ {timestamp} ({m.group(1)} success, {m.group(2)} failed)"
                elif operation == 'weekly_summary' and 'No notification channels configured' in combined_text:
                    result_msg = f"{op_label} OK @ {timestamp} (no channels configured)"
            else:
                lock_conflict = (
                    operation == 'scan'
                    and scheduled
                    and 'Could not set lock on file' in combined_text
                )
                if lock_conflict:
                    retry_at = datetime.now() + timedelta(minutes=SCAN_LOCK_RETRY_MINUTES)
                    STATE['scheduled_time'] = retry_at
                    _persist_scan_schedule_locked()
                    result_msg = (
                        f"Scan deferred @ {timestamp} "
                        f"(DB busy; retry {retry_at.strftime('%Y-%m-%d %H:%M')})"
                    )
                else:
                    hint = ''
                    for line in reversed([ln.strip() for ln in combined_text.splitlines() if ln.strip()]):
                        if line:
                            hint = line[:100]
                            break
                    result_msg = f"{op_label} FAILED @ {timestamp}" + (f" ({hint})" if hint else "")
            STATE[f'{operation}_last_result'] = result_msg
            STATE['last_result'] = result_msg
        
        with PROCESS_LOCK:
            PROCESSES[operation] = None
            
    except Exception as e:
        with STATE_LOCK:
            STATE[f'{operation}_running'] = False
            STATE[f'{operation}_last_result'] = f"Error: {e}"
            STATE['last_result'] = f"Error: {e}"
        with PROCESS_LOCK:
            PROCESSES[operation] = None


@app.route('/train_schedule', methods=['POST', 'DELETE'])
def train_schedule():
    if request.method == 'DELETE':
        with STATE_LOCK:
            STATE['train_scheduled_time'] = None
        return jsonify({'status': 'cancelled'})
    
    data = request.get_json()
    time_str = data.get('time', '21:00')
    interval_days = data.get('days', 10)
    allowed_days = data.get('allowed_days', [0,1,2,3,4,5,6])
    
    hour, minute = map(int, time_str.split(':'))
    
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    if target <= now:
        target += timedelta(days=1)
        
    while target.weekday() not in allowed_days:
        target += timedelta(days=1)
    
    with STATE_LOCK:
        STATE['train_scheduled_time'] = target
        STATE['train_interval_days'] = interval_days
        STATE['train_days'] = allowed_days
    
    def train_scheduler_loop():
        while True:
            with STATE_LOCK:
                if STATE['train_scheduled_time'] is None:
                    return
                
                now = datetime.now()
                target = STATE['train_scheduled_time']
                interval = STATE['train_interval_days']
                window = STATE['schedule_window_minutes']
                allow = STATE.get('train_days', [0,1,2,3,4,5,6])
                
                diff = (now - target).total_seconds() / 60
                
                if now >= target:
                    if abs(diff) <= window:
                        if not STATE.get('train_running'):
                            STATE['train_running'] = True
                            
                            next_run = get_next_valid_time(target, interval, allow)
                            STATE['train_scheduled_time'] = next_run
                            STATE['last_result'] = f"Train triggered @ {now.strftime('%Y-%m-%d %H:%M')}"
                            
                            threading.Thread(target=lambda: run_op_internal('train'), daemon=True).start()
                            
                    elif diff > window:
                        next_run = get_next_valid_time(target, interval, allow)
                        STATE['train_scheduled_time'] = next_run
                        STATE['last_result'] = f"Missed train, next: {next_run.strftime('%Y-%m-%d %H:%M')}"
            
            time.sleep(30)
    
    threading.Thread(target=train_scheduler_loop, daemon=True).start()
    return jsonify({'status': 'scheduled', 'time': target.strftime('%Y-%m-%d %H:%M'), 'interval': interval_days})



@app.route('/run/qual', methods=['POST'])
def run_qual_route():
    """Run qualitative feature update."""
    data = request.get_json() or {}
    chunk_size = data.get('chunk_size', 20)
    
    with STATE_LOCK:
        if STATE.get('qual_running'):
            return jsonify({'error': 'Qual already running'}), 400
        STATE['qual_running'] = True
        STATE['qual_chunk_size'] = chunk_size
    
    threading.Thread(target=lambda: run_op_internal('qual'), daemon=True).start()
    return jsonify({'status': 'started', 'chunk_size': chunk_size})


@app.route('/qual_schedule', methods=['POST', 'DELETE'])
def qual_schedule():
    if request.method == 'DELETE':
        with STATE_LOCK:
            STATE['qual_scheduled_time'] = None
        return jsonify({'status': 'cancelled'})
    
    data = request.get_json()
    chunk_size = data.get('chunk_size', 20)
    interval_hours = data.get('interval_hours', 4)
    
    now = datetime.now()
    target = now + timedelta(hours=interval_hours)
    
    with STATE_LOCK:
        STATE['qual_scheduled_time'] = target
        STATE['qual_chunk_size'] = chunk_size
        STATE['qual_interval_hours'] = interval_hours
    
    def qual_scheduler_loop():
        while True:
            with STATE_LOCK:
                if STATE['qual_scheduled_time'] is None:
                    return
                
                now = datetime.now()
                target = STATE['qual_scheduled_time']
                interval = STATE['qual_interval_hours']
                window = STATE['schedule_window_minutes']
                
                diff = (now - target).total_seconds() / 60
                
                if now >= target:
                    if abs(diff) <= window:
                        if not STATE.get('qual_running'):
                            STATE['qual_running'] = True
                            
                            next_run = now + timedelta(hours=interval)
                            STATE['qual_scheduled_time'] = next_run
                            STATE['last_result'] = f"Qual triggered @ {now.strftime('%Y-%m-%d %H:%M')}"
                            
                            threading.Thread(target=lambda: run_op_internal('qual'), daemon=True).start()
                            
                    elif diff > window:
                        next_run = now + timedelta(hours=interval)
                        STATE['qual_scheduled_time'] = next_run
                        STATE['last_result'] = f"Missed qual, next: {next_run.strftime('%Y-%m-%d %H:%M')}"
            
            time.sleep(30)
    
    threading.Thread(target=qual_scheduler_loop, daemon=True).start()
    return jsonify({'status': 'scheduled', 'next': target.strftime('%Y-%m-%d %H:%M'), 'interval_hours': interval_hours})


@app.route('/snapshot_schedule', methods=['POST', 'DELETE'])
def snapshot_schedule():
    if request.method == 'DELETE':
        with STATE_LOCK:
            STATE['snapshot_scheduled_time'] = None
        return jsonify({'status': 'cancelled'})

    data = request.get_json(silent=True) or {}
    interval_minutes = _safe_int(data.get('interval_minutes'), STATE.get('snapshot_interval_minutes', 45)) or 45
    recent_days = _safe_int(data.get('recent_days'), STATE.get('snapshot_recent_days', 30)) or 30
    interval_minutes = max(5, min(720, interval_minutes))
    recent_days = max(1, min(365, recent_days))
    target = datetime.now() + timedelta(minutes=interval_minutes)

    with STATE_LOCK:
        STATE['snapshot_interval_minutes'] = interval_minutes
        STATE['snapshot_recent_days'] = recent_days
        STATE['snapshot_scheduled_time'] = target

    _start_snapshot_scheduler_once()
    return jsonify(
        {
            'status': 'scheduled',
            'next': target.strftime('%Y-%m-%d %H:%M'),
            'interval_minutes': interval_minutes,
            'recent_days': recent_days,
        }
    )


@app.route('/news_schedule', methods=['POST', 'DELETE'])
def news_schedule():
    if request.method == 'DELETE':
        with STATE_LOCK:
            STATE['news_scheduled_time'] = None
        return jsonify({'status': 'cancelled'})

    data = request.get_json() or {}
    morning_time = data.get('morning_time', '08:30')
    evening_time = data.get('evening_time', '16:30')
    days = _sanitize_days(data.get('days', [0, 1, 2, 3, 4]), fallback=[0, 1, 2, 3, 4])
    morning_h, morning_m = _parse_hour_minute(morning_time, '08:30')
    evening_h, evening_m = _parse_hour_minute(evening_time, '16:30')

    # Determine next run: whichever is soonest (morning or evening today, or next morning)
    now = datetime.now()
    morning_today = now.replace(hour=morning_h, minute=morning_m, second=0, microsecond=0)
    evening_today = now.replace(hour=evening_h, minute=evening_m, second=0, microsecond=0)

    if now < morning_today and now.weekday() in days:
        target = morning_today
    elif now < evening_today and now.weekday() in days:
        target = evening_today
    else:
        target = _next_run_for_days(morning_h, morning_m, days, now=now + timedelta(minutes=1))

    with STATE_LOCK:
        STATE['news_morning_time'] = f"{morning_h:02d}:{morning_m:02d}"
        STATE['news_evening_time'] = f"{evening_h:02d}:{evening_m:02d}"
        STATE['news_days'] = days
        STATE['news_scheduled_time'] = target

    _start_news_scheduler_once()
    return jsonify({
        'status': 'scheduled',
        'next': target.strftime('%Y-%m-%d %H:%M'),
        'morning_time': f"{morning_h:02d}:{morning_m:02d}",
        'evening_time': f"{evening_h:02d}:{evening_m:02d}",
        'days': days,
    })


@app.route('/audit_schedule', methods=['POST', 'DELETE'])
def audit_schedule():
    if request.method == 'DELETE':
        with STATE_LOCK:
            STATE['audit_scheduled_time'] = None
        return jsonify({'status': 'cancelled'})

    data = request.get_json() or {}
    time_str = data.get('time', DASHBOARD_AUDIT_DEFAULT_TIME)
    days = _sanitize_days(data.get('days', [0, 1, 2, 3, 4]), fallback=[0, 1, 2, 3, 4])
    hour, minute = _parse_hour_minute(time_str, DASHBOARD_AUDIT_DEFAULT_TIME)
    target = _next_run_for_days(hour, minute, days)

    with STATE_LOCK:
        STATE['audit_scheduled_time'] = target
        STATE['audit_time'] = f"{hour:02d}:{minute:02d}"
        STATE['audit_days'] = days

    _start_audit_scheduler_once()
    return jsonify({
        'status': 'scheduled',
        'next': target.strftime('%Y-%m-%d %H:%M'),
        'days': days,
    })


@app.route('/weekly_summary_schedule', methods=['POST', 'DELETE'])
def weekly_summary_schedule():
    if request.method == 'DELETE':
        with STATE_LOCK:
            STATE['weekly_summary_scheduled_time'] = None
        return jsonify({'status': 'cancelled'})

    data = request.get_json() or {}
    time_str = data.get('time', DASHBOARD_WEEKLY_SUMMARY_DEFAULT_TIME)
    day = _sanitize_days([data.get('day', DASHBOARD_WEEKLY_SUMMARY_DEFAULT_DAY)], fallback=[4])[0]
    hour, minute = _parse_hour_minute(time_str, DASHBOARD_WEEKLY_SUMMARY_DEFAULT_TIME)
    target = _next_run_for_days(hour, minute, [day])

    with STATE_LOCK:
        STATE['weekly_summary_day'] = day
        STATE['weekly_summary_time'] = f"{hour:02d}:{minute:02d}"
        STATE['weekly_summary_scheduled_time'] = target

    _start_weekly_summary_scheduler_once()
    return jsonify({
        'status': 'scheduled',
        'next': target.strftime('%Y-%m-%d %H:%M'),
        'day': day,
    })


# Restore persisted scan schedule and ensure schedulers are active on startup.
_load_scan_schedule_state()
_start_scan_scheduler_once()
_start_news_scheduler_once()
_start_audit_scheduler_once()
_start_weekly_summary_scheduler_once()
_start_snapshot_scheduler_once()


if __name__ == '__main__':
    print("=" * 50)
    print("Market Analyzer Dashboard")
    print("=" * 50)
    print(f"Local URL: http://127.0.0.1:5050")
    print(f"")
    print(f"To access remotely via SSH tunnel:")
    print(f"  ssh -L 5050:127.0.0.1:5050 {SSH_TUNNEL_TARGET}")
    print(f"  Then open: http://localhost:5050")
    print("=" * 50)
    
    app.run(host='127.0.0.1', port=5050, debug=False)
