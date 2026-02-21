# MktML Market Analysis Report - YYYY-MM-DD

## Agent Morning Snapshot
<!-- OPENCLAW:SUMMARY:START -->
- report_date: YYYY-MM-DD
- generated_at: YYYY-MM-DD HH:MM
- regime: ⚪ NEUTRAL
- signal_bias: balanced
- buys: 12
- sells: 9
- exit_alerts: none
- data_health: coverage=100.0%, stale=0, fetch_failures=0
- top_buys: AAA,BBB,CCC
- top_sells: XXX,YYY,ZZZ
- actions: review_top_buys
<!-- OPENCLAW:SUMMARY:END -->

### Agent Payload (JSON)
<!-- OPENCLAW:JSON:START -->
```json
{
  "schema_version": "market_report_agent_v1",
  "report_date": "YYYY-MM-DD",
  "generated_at": "YYYY-MM-DD HH:MM",
  "market": {
    "macro_regime": "NEUTRAL",
    "signal_bias": "balanced"
  },
  "counts": {
    "tickers_scanned": 1000,
    "buys": 12,
    "sells": 9
  },
  "alerts": {
    "exit_alerts": []
  },
  "top_buys": [
    {"ticker": "AAA", "price": 123.45}
  ],
  "top_sells": [
    {"ticker": "XXX", "price": 67.89}
  ],
  "actions": ["review_top_buys"]
}
```
<!-- OPENCLAW:JSON:END -->
