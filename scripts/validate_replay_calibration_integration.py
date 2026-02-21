#!/usr/bin/env python3
"""Run integrated replay reproducibility and calibration/threshold validation."""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from config import (
    CALIBRATION_ARTIFACT_DIR,
    MODEL_THRESHOLDS_FILE,
    PROB_CALIBRATION_FILE_TEMPLATE,
)

try:
    from scanner import run_replay_reproducibility_check
    SCANNER_HELPER_AVAILABLE = True
except Exception:  # pragma: no cover - defensive for damaged env
    run_replay_reproducibility_check = None
    SCANNER_HELPER_AVAILABLE = False

HORIZONS = (5, 10, 30)
REPLAY_VALIDATION_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "validate_replay_reproducibility.py")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate replay reproducibility + calibration integration.",
    )
    parser.add_argument("--start", required=True, help="Replay start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Replay end date (YYYY-MM-DD)")
    parser.add_argument(
        "--replays",
        type=int,
        default=2,
        help="Number of replay runs for reproducibility (minimum 2)",
    )
    parser.add_argument(
        "--replay-limit",
        type=int,
        default=None,
        help="Optional ticker limit for replay scan",
    )
    parser.add_argument(
        "--flush-rows",
        type=int,
        default=5000,
        help="Replay upsert flush batch size",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional explicit markdown report path",
    )
    return parser.parse_args()


def _load_json_file(path: str) -> Tuple[bool, Any, Optional[str]]:
    if not os.path.exists(path):
        return False, None, "missing"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return True, json.load(f), None
    except Exception as exc:  # pragma: no cover - depends on environment file health
        return False, None, str(exc)


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _artifact_path(horizon: int) -> str:
    return os.path.join(
        CALIBRATION_ARTIFACT_DIR,
        PROB_CALIBRATION_FILE_TEMPLATE.format(horizon=horizon),
    )


def _validate_calibration_for_horizon(horizon: int) -> Dict[str, Any]:
    path = _artifact_path(int(horizon))
    result = {
        "horizon": horizon,
        "path": path,
        "exists": os.path.exists(path),
        "parseable": False,
        "horizon_match": False,
        "status": "missing",
    }

    ok, payload, error = _load_json_file(path)
    if not ok:
        result["status"] = f"unreadable:{error}"
        return result

    if not isinstance(payload, dict):
        result["status"] = "invalid_root"
        return result

    result["parseable"] = isinstance(payload.get("bin_edges"), list) and isinstance(
        payload.get("bin_calibrated"), list
    )
    if not result["parseable"]:
        result["status"] = "invalid_payload"
        return result

    expected_horizon = payload.get("horizon")
    try:
        result["horizon_match"] = int(expected_horizon) == int(horizon)
    except (TypeError, ValueError):
        result["horizon_match"] = False

    if not result["horizon_match"]:
        result["status"] = "horizon_mismatch"
        return result

    result["status"] = "ok"
    return result


def _validate_calibration_artifacts() -> Dict[str, Any]:
    artifact_checks: Dict[str, Dict[str, Any]] = {}
    all_ok = True
    for horizon in HORIZONS:
        check = _validate_calibration_for_horizon(horizon)
        artifact_checks[str(horizon)] = check
        all_ok = all_ok and check["parseable"] and check["horizon_match"] and check["status"] == "ok"
    return {
        "passed": all_ok,
        "details": artifact_checks,
    }


def _validate_thresholds() -> Dict[str, Any]:
    success, payload, error = _load_json_file(MODEL_THRESHOLDS_FILE)
    details: Dict[str, Dict[str, Any]] = {}
    checks_ok = True

    if not success or not isinstance(payload, dict):
        status = "unreadable" if not success else "invalid_root"
        for horizon in HORIZONS:
            details[str(horizon)] = {"status": status, "exists": False}
        return {"passed": False, "error": error, "details": details}

    for horizon in HORIZONS:
        key = str(horizon)
        entry = payload.get(key)
        check = {
            "status": "ok",
            "exists": isinstance(entry, dict),
            "buy": None,
            "sell": None,
        }
        if not isinstance(entry, dict):
            check["status"] = "missing_entry"
            checks_ok = False
        else:
            buy = _coerce_float(entry.get("buy"))
            sell = _coerce_float(entry.get("sell"))
            check["buy"] = buy
            check["sell"] = sell
            if buy is None or sell is None:
                check["status"] = "invalid_entry"
                checks_ok = False
        details[key] = check

    return {"passed": checks_ok, "details": details}


def _run_replay_check(args: argparse.Namespace) -> Dict[str, Any]:
    command_path = REPLAY_VALIDATION_SCRIPT
    command_exists = os.path.exists(command_path)
    if not SCANNER_HELPER_AVAILABLE or run_replay_reproducibility_check is None:
        return {
            "passed": False,
            "command_path": command_path,
            "command_exists": command_exists,
            "status": "helper_unavailable",
            "status_text": "run_replay_reproducibility_check import failed",
        }

    if args.replays < 2:
        return {
            "passed": False,
            "command_path": command_path,
            "command_exists": command_exists,
            "status": "invalid_args",
            "status_text": "replays must be >= 2",
        }

    try:
        result = run_replay_reproducibility_check(
            start_date=args.start,
            end_date=args.end,
            replay_runs=args.replays,
            max_tickers=args.replay_limit,
            flush_rows=args.flush_rows,
        )
        passed = result.get("status") == "PASS"
        return {
            "passed": bool(passed),
            "command_path": command_path,
            "command_exists": command_exists,
            "status": result.get("status"),
            "status_text": "ok" if passed else result.get("fatal_error") or "mismatch_detected",
            "summary": {
                "start_date": result.get("start_date"),
                "end_date": result.get("end_date"),
                "replay_runs": result.get("replay_runs"),
                "rows_compared": result.get("rows_compared", 0),
                "mismatch_count": result.get("mismatch_count", 0),
                "replay_report_path": result.get("report_path"),
            },
        }
    except Exception as exc:
        return {
            "passed": False,
            "command_path": command_path,
            "command_exists": command_exists,
            "status": "execution_failed",
            "status_text": str(exc),
        }


def _default_report_path() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(REPORT_DIR, f"integration_validation_{timestamp}.md")


def _write_report(summary: Dict[str, Any], path: Optional[str]) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.abspath(path or _default_report_path())
    replay = summary.get("replay", {})
    calib = summary.get("calibration", {})
    thresholds = summary.get("thresholds", {})

    lines: List[str] = []
    lines.append("# Replay + Calibration Integration Check")
    lines.append("")
    lines.append(f"- Timestamp: {summary.get('timestamp')}")
    lines.append(f"- Status: {summary.get('status')}")
    lines.append("")
    lines.append("## Replay")
    lines.append(f"- Command path: {replay.get('command_path')}")
    lines.append(f"- Command exists: {replay.get('command_exists')}")
    lines.append(f"- Status: {replay.get('status')}")
    lines.append(f"- Passed: {replay.get('passed')}")
    lines.append(f"- Status text: {replay.get('status_text')}")
    rows = replay.get("summary") or {}
    lines.append(f"- Rows compared: {rows.get('rows_compared', 0)}")
    lines.append(f"- Mismatch count: {rows.get('mismatch_count', 0)}")
    lines.append(f"- Replay report path: {rows.get('replay_report_path')}")
    lines.append("")

    lines.append("## Calibration artifacts")
    details = calib.get("details", {})
    for horizon in map(str, HORIZONS):
        check = details.get(horizon, {})
        lines.append(
            f"- {horizon}d: ok={check.get('status') == 'ok'}, "
            f"exists={check.get('exists')}, horizon_match={check.get('horizon_match')}"
        )
    lines.append("")

    lines.append("## Thresholds")
    th = thresholds.get("details", {})
    for horizon in map(str, HORIZONS):
        check = th.get(horizon, {})
        lines.append(
            f"- {horizon}d: exists={check.get('exists')}, "
            f"status={check.get('status')}, "
            f"buy={check.get('buy')}, sell={check.get('sell')}"
        )
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


def main() -> int:
    args = _parse_args()
    calibration = _validate_calibration_artifacts()
    thresholds = _validate_thresholds()
    replay = _run_replay_check(args)

    status_pass = bool(replay.get("passed")) and bool(calibration.get("passed")) and bool(thresholds.get("passed"))
    summary: Dict[str, Any] = {
        "status": "PASS" if status_pass else "FAIL",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "replay": replay,
        "calibration": calibration,
        "thresholds": thresholds,
    }

    summary["report_path"] = _write_report(summary, args.report)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
