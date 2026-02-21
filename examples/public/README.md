# MktML Public Examples

This folder contains publish-safe starter files for MktML.

## Files
- `config.public.example.py`: full sanitized config template (no embedded secrets, no hardcoded holdings/watchlist).
- `.env.example`: environment variable template for credentials and runtime knobs.
- `market_report.sample.md`: publish-safe example output structure including `OPENCLAW` parsing markers.
- `launchd/`: generic macOS scheduler/service templates without host-specific identifiers.

## Typical setup
1. Copy `examples/public/config.public.example.py` to `config.py`.
2. Copy `examples/public/.env.example` to `.env` (or export variables in shell).
3. Fill in provider API keys.
4. Edit `PORTFOLIO_HOLDINGS` and `WATCHLIST` in `config.py`.
