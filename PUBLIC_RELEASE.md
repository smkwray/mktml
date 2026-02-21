# MktML Open-Source Release Guide

Use this checklist before publishing this repository.

## What is now excluded by default
The project-level `.gitignore` excludes:
- internal workflow workspace (`prompt/`)
- runtime artifacts (`data/`, `models/`, `reports/`, `logs/`, status/log files)
- runtime config and secrets (`config.py`, `.env*`)
- archives/history (`old/`, `docs-archive/`, `*archive*/`)
- host-specific launchd/ops files

## Public-friendly starter files
- `examples/public/config.public.example.py`
- `examples/public/.env.example`
- `examples/public/README.md`
- `examples/public/market_report.sample.md`
- `examples/public/launchd/` (generic launchd templates)

## Never publish directly
- `config.py` (runtime credentials and deployment-specific settings)
- `prompt/` (internal planning/workflow notes)
- `reports/market_report_*.md` and `reports-archive/market_report_*.md` (generated strategy output)
- `logs/` and `logs/spark/` (runtime traces and machine details)
- `data/config_history.jsonl` and other runtime state/history files

## Release checklist
1. Confirm git root is this project folder (not a parent directory):
   - `git rev-parse --show-toplevel`
2. Confirm no sensitive files are staged:
   - `git status --short -- .`
3. Confirm the publish-safe config template exists and is sanitized:
   - `examples/public/config.public.example.py`
4. Confirm API key defaults in public files are blank/placeholders.
5. Run a quick compile check:
   - `python -m py_compile src/*.py`
6. Optional secret scan:
   - `rg -n "(API_KEY|SECRET|TOKEN|WEBHOOK|PASSWORD)" examples/public`
7. Optional sensitive-string scan in public docs/examples:
   - `rg -n "(/Users/|@\\d+\\.\\d+\\.\\d+\\.\\d+|com\\.[a-z0-9_-]+\\.[a-z0-9_.-]*|PORTFOLIO_HOLDINGS\\s*=\\s*\\[|WATCHLIST\\s*=\\s*\\[)" README.md PUBLIC_RELEASE.md examples/public`
8. Enable and run push guard:
   - `git config core.hooksPath .githooks`
   - `scripts/public_push_guard.sh`

## Notes
- For runtime use, maintain a project-local `config.py`; it is intentionally ignored.
- If you need to share sample outputs, create redacted files under `examples/public/` rather than sharing `reports/` directly.
- If any credentials were ever committed in history, rotate those keys before publishing.
