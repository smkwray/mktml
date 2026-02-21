#!/usr/bin/env python3
"""Generate a read-only analytics snapshot from DuckDB."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import duckdb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from config import HORIZON_TARGETS  # noqa: E402
from storage import DB_PATH  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate project analytics snapshot (Markdown + JSON).",
    )
    parser.add_argument(
        "--db-path",
        default=DB_PATH,
        help="DuckDB path (default: storage.DB_PATH).",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Optional explicit markdown output path.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional explicit JSON output path.",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=30,
        help="Lookback window for recent recommendation signal mix.",
    )
    return parser.parse_args()


def _table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    row = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE lower(table_name) = lower(?)
        """,
        [table_name],
    ).fetchone()
    return bool(row and int(row[0]) > 0)


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _rows_to_dicts(columns: List[str], rows: List[tuple]) -> List[Dict[str, Any]]:
    return [{columns[i]: row[i] for i in range(len(columns))} for row in rows]


def _gather_snapshot(con: duckdb.DuckDBPyConnection, recent_days: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "db_path": "",
        "tables_present": {},
    }

    has_price = _table_exists(con, "price_history")
    has_recs = _table_exists(con, "recommendation_history")
    has_preds = _table_exists(con, "model_predictions")
    out["tables_present"] = {
        "price_history": has_price,
        "recommendation_history": has_recs,
        "model_predictions": has_preds,
    }

    if has_price:
        price_row = con.execute(
            """
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT upper(ticker)) AS tickers,
                MIN(date) AS min_date,
                MAX(date) AS max_date
            FROM price_history
            """
        ).fetchone()
        out["price_history"] = {
            "rows": _safe_int(price_row[0]),
            "tickers": _safe_int(price_row[1]),
            "min_date": _safe_str(price_row[2]),
            "max_date": _safe_str(price_row[3]),
        }
    else:
        out["price_history"] = {}

    if has_recs:
        rec_total = con.execute(
            "SELECT COUNT(*), MAX(date) FROM recommendation_history"
        ).fetchone()
        rec_mix_rows = con.execute(
            """
            SELECT signal_type, COUNT(*) AS rows
            FROM recommendation_history
            WHERE date >= current_date - (? * INTERVAL '1 day')
            GROUP BY signal_type
            ORDER BY rows DESC, signal_type
            """,
            [max(1, int(recent_days))],
        ).fetchall()
        out["recommendations"] = {
            "rows_total": _safe_int(rec_total[0]),
            "max_date": _safe_str(rec_total[1]),
            "recent_days": max(1, int(recent_days)),
            "recent_signal_mix": [
                {"signal_type": _safe_str(sig), "rows": _safe_int(cnt)}
                for sig, cnt in rec_mix_rows
            ],
        }
    else:
        out["recommendations"] = {}

    if not has_preds:
        out["model_predictions"] = {}
        return out

    preds_total = con.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT model_version_hash) AS model_hashes,
            MAX(created_at) AS max_created_at
        FROM model_predictions
        """
    ).fetchone()

    latest_hash_row = con.execute(
        """
        SELECT model_version_hash
        FROM model_predictions
        WHERE model_version_hash IS NOT NULL
          AND model_version_hash <> ''
        ORDER BY created_at DESC
        LIMIT 1
        """
    ).fetchone()
    latest_hash = _safe_str(latest_hash_row[0]) if latest_hash_row else ""

    by_horizon_rows = []
    if latest_hash:
        by_horizon_rows = con.execute(
            """
            SELECT
                horizon,
                COUNT(*) AS rows,
                COUNT(DISTINCT asof_date) AS dates,
                COUNT(DISTINCT upper(ticker)) AS tickers,
                MIN(asof_date) AS min_asof_date,
                MAX(asof_date) AS max_asof_date,
                MAX(created_at) AS max_created_at
            FROM model_predictions
            WHERE model_version_hash = ?
            GROUP BY horizon
            ORDER BY horizon
            """,
            [latest_hash],
        ).fetchall()

    throughput_windows = [5, 15, 60]
    throughput_rows_per_min: Dict[str, float] = {}
    for minutes in throughput_windows:
        if latest_hash:
            cnt_row = con.execute(
                f"""
                SELECT COUNT(*)
                FROM model_predictions
                WHERE model_version_hash = ?
                  AND created_at > now() - interval '{minutes} minutes'
                """,
                [latest_hash],
            ).fetchone()
            rows = _safe_int(cnt_row[0])
        else:
            rows = 0
        throughput_rows_per_min[f"{minutes}m"] = rows / float(minutes)

    signal_mix_24h_rows = []
    if latest_hash:
        signal_mix_24h_rows = con.execute(
            """
            SELECT signal_type, COUNT(*) AS rows
            FROM model_predictions
            WHERE model_version_hash = ?
              AND created_at > now() - interval '24 hours'
            GROUP BY signal_type
            ORDER BY rows DESC, signal_type
            """,
            [latest_hash],
        ).fetchall()

    prob_stats_rows = []
    if latest_hash:
        prob_stats_rows = con.execute(
            """
            SELECT
                horizon,
                COUNT(*) AS rows,
                AVG(proba_raw) AS mean_raw,
                AVG(proba_cal) AS mean_cal,
                STDDEV_POP(proba_cal) AS std_cal,
                quantile_cont(proba_cal, 0.10) AS p10_cal,
                quantile_cont(proba_cal, 0.50) AS p50_cal,
                quantile_cont(proba_cal, 0.90) AS p90_cal
            FROM model_predictions
            WHERE model_version_hash = ?
            GROUP BY horizon
            ORDER BY horizon
            """,
            [latest_hash],
        ).fetchall()

    calib_quality_rows = []
    if latest_hash and has_price:
        t5 = float(HORIZON_TARGETS.get(5, 0.0) or 0.0)
        t10 = float(HORIZON_TARGETS.get(10, 0.0) or 0.0)
        t30 = float(HORIZON_TARGETS.get(30, 0.0) or 0.0)
        calib_quality_rows = con.execute(
            """
            WITH ranked_prices AS (
                SELECT
                    upper(ticker) AS ticker,
                    date,
                    close,
                    row_number() OVER (PARTITION BY upper(ticker) ORDER BY date) AS rn
                FROM price_history
            ),
            joined AS (
                SELECT
                    mp.horizon,
                    mp.proba_raw,
                    mp.proba_cal,
                    ((p1.close / p0.close) - 1.0) AS fwd_return,
                    CASE
                        WHEN mp.horizon = 5 AND ((p1.close / p0.close) - 1.0) >= ? THEN 1
                        WHEN mp.horizon = 10 AND ((p1.close / p0.close) - 1.0) >= ? THEN 1
                        WHEN mp.horizon = 30 AND ((p1.close / p0.close) - 1.0) >= ? THEN 1
                        ELSE 0
                    END AS y_true
                FROM model_predictions mp
                JOIN ranked_prices p0
                  ON p0.ticker = upper(mp.ticker)
                 AND p0.date = mp.asof_date
                JOIN ranked_prices p1
                  ON p1.ticker = p0.ticker
                 AND p1.rn = p0.rn + mp.horizon
                WHERE mp.model_version_hash = ?
                  AND mp.horizon IN (5, 10, 30)
                  AND p0.close IS NOT NULL
                  AND p1.close IS NOT NULL
                  AND p0.close > 0
            )
            SELECT
                horizon,
                COUNT(*) AS rows,
                AVG(CAST(y_true AS DOUBLE)) AS hit_rate,
                AVG(POWER(proba_raw - CAST(y_true AS DOUBLE), 2)) AS brier_raw,
                AVG(POWER(proba_cal - CAST(y_true AS DOUBLE), 2)) AS brier_cal,
                AVG(proba_cal) AS mean_proba_cal
            FROM joined
            GROUP BY horizon
            ORDER BY horizon
            """,
            [t5, t10, t30, latest_hash],
        ).fetchall()

    out["model_predictions"] = {
        "rows_total": _safe_int(preds_total[0]),
        "distinct_model_hashes": _safe_int(preds_total[1]),
        "max_created_at": _safe_str(preds_total[2]),
        "latest_model_hash": latest_hash,
        "latest_hash_by_horizon": [
            {
                "horizon": _safe_int(row[0]),
                "rows": _safe_int(row[1]),
                "dates": _safe_int(row[2]),
                "tickers": _safe_int(row[3]),
                "min_asof_date": _safe_str(row[4]),
                "max_asof_date": _safe_str(row[5]),
                "max_created_at": _safe_str(row[6]),
            }
            for row in by_horizon_rows
        ],
        "throughput_rows_per_min": throughput_rows_per_min,
        "signal_mix_last_24h": [
            {"signal_type": _safe_str(row[0]), "rows": _safe_int(row[1])}
            for row in signal_mix_24h_rows
        ],
        "probability_distribution_by_horizon": [
            {
                "horizon": _safe_int(row[0]),
                "rows": _safe_int(row[1]),
                "mean_raw": float(row[2]) if row[2] is not None else None,
                "mean_cal": float(row[3]) if row[3] is not None else None,
                "std_cal": float(row[4]) if row[4] is not None else None,
                "p10_cal": float(row[5]) if row[5] is not None else None,
                "p50_cal": float(row[6]) if row[6] is not None else None,
                "p90_cal": float(row[7]) if row[7] is not None else None,
            }
            for row in prob_stats_rows
        ],
        "label_quality_by_horizon": [
            {
                "horizon": _safe_int(row[0]),
                "rows": _safe_int(row[1]),
                "hit_rate": float(row[2]) if row[2] is not None else None,
                "brier_raw": float(row[3]) if row[3] is not None else None,
                "brier_cal": float(row[4]) if row[4] is not None else None,
                "mean_proba_cal": float(row[5]) if row[5] is not None else None,
            }
            for row in calib_quality_rows
        ],
    }
    return out


def _render_markdown(snapshot: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Analytics Snapshot")
    lines.append("")
    lines.append(f"- Generated (UTC): `{snapshot.get('generated_at_utc', '')}`")
    lines.append(f"- DB path: `{snapshot.get('db_path', '')}`")
    lines.append("")

    lines.append("## Table Presence")
    table_flags = snapshot.get("tables_present", {})
    for table_name in ("price_history", "recommendation_history", "model_predictions"):
        lines.append(f"- `{table_name}`: `{bool(table_flags.get(table_name, False))}`")
    lines.append("")

    price = snapshot.get("price_history", {})
    lines.append("## Price History")
    lines.append(f"- Rows: `{_safe_int(price.get('rows'))}`")
    lines.append(f"- Tickers: `{_safe_int(price.get('tickers'))}`")
    lines.append(f"- Date range: `{_safe_str(price.get('min_date'))}` -> `{_safe_str(price.get('max_date'))}`")
    lines.append("")

    recs = snapshot.get("recommendations", {})
    lines.append("## Recommendations")
    lines.append(f"- Total rows: `{_safe_int(recs.get('rows_total'))}`")
    lines.append(f"- Max date: `{_safe_str(recs.get('max_date'))}`")
    lines.append(f"- Recent window (days): `{_safe_int(recs.get('recent_days'))}`")
    mix = recs.get("recent_signal_mix", [])
    if mix:
        lines.append("- Recent signal mix:")
        for row in mix:
            lines.append(f"  - `{_safe_str(row.get('signal_type'))}`: `{_safe_int(row.get('rows'))}`")
    else:
        lines.append("- Recent signal mix: `n/a`")
    lines.append("")

    preds = snapshot.get("model_predictions", {})
    lines.append("## Model Predictions")
    lines.append(f"- Total rows: `{_safe_int(preds.get('rows_total'))}`")
    lines.append(f"- Distinct model hashes: `{_safe_int(preds.get('distinct_model_hashes'))}`")
    lines.append(f"- Latest created_at: `{_safe_str(preds.get('max_created_at'))}`")
    lines.append(f"- Latest model hash: `{_safe_str(preds.get('latest_model_hash'))}`")
    lines.append("")

    lines.append("## Replay Coverage (Latest Hash)")
    latest_cov = preds.get("latest_hash_by_horizon", [])
    if latest_cov:
        lines.append("| Horizon | Rows | Dates | Tickers | Min As-Of | Max As-Of | Max Created |")
        lines.append("|---:|---:|---:|---:|---|---|---|")
        for row in latest_cov:
            lines.append(
                "| {h} | {rows} | {dates} | {tickers} | {min_asof} | {max_asof} | {max_created} |".format(
                    h=_safe_int(row.get("horizon")),
                    rows=_safe_int(row.get("rows")),
                    dates=_safe_int(row.get("dates")),
                    tickers=_safe_int(row.get("tickers")),
                    min_asof=_safe_str(row.get("min_asof_date")),
                    max_asof=_safe_str(row.get("max_asof_date")),
                    max_created=_safe_str(row.get("max_created_at")),
                )
            )
    else:
        lines.append("No replay coverage data found for latest hash.")
    lines.append("")

    lines.append("## Throughput (Latest Hash)")
    tput = preds.get("throughput_rows_per_min", {})
    if tput:
        for key in ("5m", "15m", "60m"):
            lines.append(f"- Rows/min `{key}`: `{_fmt_float(tput.get(key), 2)}`")
    else:
        lines.append("- `n/a`")
    lines.append("")

    lines.append("## Signal Mix (Last 24h, Latest Hash)")
    sig_mix = preds.get("signal_mix_last_24h", [])
    if sig_mix:
        for row in sig_mix:
            lines.append(f"- `{_safe_str(row.get('signal_type'))}`: `{_safe_int(row.get('rows'))}`")
    else:
        lines.append("- `n/a`")
    lines.append("")

    lines.append("## Probability Distribution by Horizon")
    prob_rows = preds.get("probability_distribution_by_horizon", [])
    if prob_rows:
        lines.append("| Horizon | Rows | Mean Raw | Mean Cal | Std Cal | P10 Cal | P50 Cal | P90 Cal |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in prob_rows:
            lines.append(
                "| {h} | {rows} | {mraw} | {mcal} | {std} | {p10} | {p50} | {p90} |".format(
                    h=_safe_int(row.get("horizon")),
                    rows=_safe_int(row.get("rows")),
                    mraw=_fmt_float(row.get("mean_raw"), 4),
                    mcal=_fmt_float(row.get("mean_cal"), 4),
                    std=_fmt_float(row.get("std_cal"), 4),
                    p10=_fmt_float(row.get("p10_cal"), 4),
                    p50=_fmt_float(row.get("p50_cal"), 4),
                    p90=_fmt_float(row.get("p90_cal"), 4),
                )
            )
    else:
        lines.append("No probability distribution rows found.")
    lines.append("")

    lines.append("## Label Quality by Horizon (Replay Join)")
    qual_rows = preds.get("label_quality_by_horizon", [])
    if qual_rows:
        lines.append("| Horizon | Rows | Hit Rate | Brier Raw | Brier Cal | Mean Proba Cal |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for row in qual_rows:
            lines.append(
                "| {h} | {rows} | {hit} | {braw} | {bcal} | {mcal} |".format(
                    h=_safe_int(row.get("horizon")),
                    rows=_safe_int(row.get("rows")),
                    hit=_fmt_float(row.get("hit_rate"), 4),
                    braw=_fmt_float(row.get("brier_raw"), 4),
                    bcal=_fmt_float(row.get("brier_cal"), 4),
                    mcal=_fmt_float(row.get("mean_proba_cal"), 4),
                )
            )
    else:
        lines.append("No matured replay label rows available yet for quality metrics.")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()

    db_path = os.path.abspath(args.db_path)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DuckDB file does not exist: {db_path}")

    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_md = os.path.abspath(args.output_md) if args.output_md else os.path.join(
        reports_dir, f"analytics_snapshot_{ts}.md"
    )
    output_json = os.path.abspath(args.output_json) if args.output_json else os.path.join(
        reports_dir, f"analytics_snapshot_{ts}.json"
    )

    con = duckdb.connect(db_path, read_only=True)
    try:
        snapshot = _gather_snapshot(con, recent_days=args.recent_days)
    finally:
        con.close()

    snapshot["db_path"] = db_path

    md = _render_markdown(snapshot)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
        f.write("\n")

    print(
        json.dumps(
            {
                "status": "ok",
                "output_md": output_md,
                "output_json": output_json,
                "latest_model_hash": (
                    (snapshot.get("model_predictions") or {}).get("latest_model_hash") or ""
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
