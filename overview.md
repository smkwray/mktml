# Market Analyzer Overview

Last updated: 2026-02-20

## What this project is
`market` is a production-oriented market analysis engine that combines:
- Technical indicator signal generation.
- Ensemble machine learning across 3 horizons (5d/10d/30d).
- Macro and dividend context features.
- Replay + calibration workflows for deterministic offline validation.
- Report generation for operational decision support.

Primary goal: produce stable, auditable BUY/HOLD/SELL recommendations with repeatable training and replay behavior.

## High-level architecture
Core modules (runtime path is under `src/`):
- `src/main.py`: CLI entrypoint and pipeline orchestrator.
- `src/dashboard.py`: Flask web control plane and operational UI.
- `src/scanner.py`: live scan flow, replay scan flow, calibration artifact builder.
- `src/ml_engine.py`: model training, feature extraction, inference, thresholds, calibration artifacts.
- `src/signals.py`: technical indicators and base signal columns.
- `src/data_loader.py`: market/fundamental data access with fallback logic.
- `src/storage.py`: DuckDB schema and read/write paths.
- `src/macro_loader.py`: FRED macro fetch/cache and as-of macro feature projection.
- `src/reporter.py`: Markdown + machine-readable report output.

Operational scripts:
- `scripts/build_model_manifest.py`: hashes model artifacts into `models/manifest.json`.
- `scripts/check_model_runtime_skew.py`: strict runtime/artifact skew validation.
- `scripts/validate_replay_calibration_integration.py`: replay + calibration integration gate.
- `scripts/validate_replay_reproducibility.py`: replay determinism checker.
- `scripts/generate_analytics_snapshot.py`: read-only DB analytics snapshot (Markdown + JSON).
- `scripts/migrate_model_artifacts.py`: migration/hardening utility for model persistence format.
- `scripts/nightly_full_with_postcheck.sh`: versioned nightly full pipeline + post-check orchestrator.
- `scripts/install_nightly_postcheck_launchd.sh`: install/update launchd recurring schedule.

## Dashboard capabilities
Run:
- `python src/dashboard.py`

Default URL:
- `http://127.0.0.1:5050`

Key capabilities:
- Start/stop/schedule scan, training, qualitative updates, audit, and weekly summary jobs.
- Edit selected config values in `config.py` from UI (portfolio/watchlist/denylist, confluence/tradability thresholds, horizon targets).
- View live run telemetry from `/api/run_monitor` (process state + CPU/memory/elapsed + analytics snapshot metrics).
- View snapshot trend sparklines (rows/min and rows total) sourced from recent analytics snapshots.
- Trigger manual snapshots or schedule recurring snapshot refreshes via `/run/snapshot` and `/snapshot_schedule`.
- Browse/open files from `reports/` and `logs/`, including inline text preview via `/api/file_preview`.
- Review config change history and rollback selected changes via `/api/config/history` and `/api/config/rollback`.

## Execution model and pipelines
Main command examples:
- Full pipeline preset: `python src/main.py --pipeline full`
- Legacy alias (deprecated): `python src/main.py --all`
- Scan only: `python src/main.py --scan`
- Train only: `python src/main.py --train-ml`
- Report only: `python src/main.py --report`
- Replay only: `python src/main.py --replay-scan --start YYYY-MM-DD --end YYYY-MM-DD`
- Calibration build only: `python src/main.py --build-calibration-artifacts --start YYYY-MM-DD --end YYYY-MM-DD`
- Artifact verification only: `python src/main.py --verify-artifacts`

Pipeline presets in `src/main.py`:
- `daily`: `scan -> report`
- `daily_auto`: `scan -> backfill -> audit -> report -> notify`
- `full`: `train -> calibrate -> verify_artifacts -> scan -> report`
- `verify`: `verify_artifacts`

## Data flow (end-to-end)
1. Data acquisition:
- `src/data_loader.py` downloads OHLCV/fundamental data using provider fallback paths.

2. Storage:
- `src/storage.py` upserts into DuckDB tables (notably `price_history`, `recommendation_history`, `model_predictions`, `fundamentals`).

3. Signal engineering:
- `src/signals.py` computes RSI/MACD/Bollinger/stochastic/SMA and related derived values.

4. ML feature extraction/inference:
- `src/ml_engine.py` transforms signal rows + macro/dividend/qualitative context into model features under a strict contract (`ML_FEATURE_CONTRACT` in `config.py`).

5. Recommendation synthesis:
- `src/scanner.py` combines multi-horizon confidence, threshold policy, confluence rules, tradability checks, and safety filters into recommendation records.

6. Reporting:
- `src/reporter.py` writes human-readable and machine-readable output under `reports/`.

## ML subsystem details
Model ensemble:
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `XGBoost` when enabled (`ENABLE_XGBOOST`)

Inference horizons:
- 5-day tactical
- 10-day swing
- 30-day trend

Training behavior highlights:
- Multi-horizon model builds.
- Walk-forward validation with date-grouped and purged split support.
- Probability calibration artifact generation.
- Threshold calibration persisted in `models/model_thresholds.json`.
- Optional asset-bucket model variants (`EQUITY`, `ETF`, `BOND`).

Model artifact safety:
- Manifest hashing in `models/manifest.json`.
- Runtime skew checks validate hash + compatibility expectations.
- Migration utilities avoid brittle pickle-only XGBoost persistence.

## Replay and calibration subsystem
Replay goal:
- Re-run inference in historical as-of mode to create reproducible `model_predictions` rows with model/data hashes.

Replay entrypoint:
- `run_replay_scan` in `src/scanner.py`.

Calibration build flow:
- `build_calibration_artifacts` in `src/scanner.py` loads replay rows from DB.
- If rows are missing for current model hash, it triggers a DB-only replay refresh.
- Per-horizon calibration artifacts are emitted under `models/calibration/`.

Why this matters:
- Decouples calibration from live API calls.
- Enables deterministic integration tests and post-train gating.

## Storage schema essentials
Core DuckDB tables (managed by `src/storage.py`):
- `price_history`: OHLCV by date/ticker.
- `recommendation_history`: generated recommendation records and outcomes.
- `model_predictions`: replay/live model probability records with hashes and metadata.
- `fundamentals`: best-effort cached fundamental fields.

The most important operational table for replay/calibration is `model_predictions`.

## Performance and threading model
Thread policy is centralized in `config.py`.

Current defaults:
- `TOTAL_CPU_CORES = os.cpu_count()`
- `RESERVED_CPU_CORES` default `4` (override with `CPU_RESERVED_CORES`)
- `AVAILABLE_CPU_CORES = TOTAL - RESERVED`
- `N_JOBS` defaults to `AVAILABLE_CPU_CORES` (override with `ML_N_JOBS`)
- `SCANNER_WORKERS` defaults from `N_JOBS` (override with `SCANNER_WORKERS`)

Oversubscription controls:
- On macOS, BLAS/OpenMP env vars default to `1` thread (`VECLIB_MAXIMUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `OMP_NUM_THREADS`).
- Keep nested parallelism bounded to avoid throughput collapse.

## Key configuration surface
All tunables live in `config.py`:
- Model/training knobs: tree counts, depth, CV splits, horizon targets.
- Feature contract: `ML_FEATURE_CONTRACT`.
- Threshold defaults and confluence rules.
- Safety/tradability policy thresholds.
- Threading/core budget controls.
- Paths for model/calibration/manifest artifacts.

## Reports and artifacts
Primary report outputs:
- `reports/live_report.md`
- `reports/model_validation.json`
- replay validation reports under `reports/`
- runtime skew reports under `reports/` when generated
- analytics snapshots: `reports/analytics_snapshot_*.md` and `reports/analytics_snapshot_*.json`

Model artifacts:
- `models/model_5d.pkl`, `models/model_10d.pkl`, `models/model_30d.pkl`
- bucket models under `models/buckets/`
- calibration artifacts under `models/calibration/`
- manifest at `models/manifest.json`

## Verification and quality checks
No single pytest suite currently drives validation. Project-standard checks are script-based:
- `python tests/verify_refactor.py`
- `python check_model_features.py`
- `python check_macro_units.py`
- `python scripts/validate_replay_reproducibility.py --start ... --end ...`
- `python scripts/validate_replay_calibration_integration.py --start ... --end ...`
- `python scripts/check_model_runtime_skew.py --strict`

## Operational guidance (remote-heavy workflows)
Heavy compute should run on the remote machine.

Canonical remote workspace:
- `<project_root>`

Recommended remote venv:
- `<venv_path>`

When tracking long runs:
- Watch pipeline logs under `logs/`.
- Use DB progress queries on `model_predictions` rather than CPU alone for replay/calibration ETA.
- Confirm completion markers in log files before assuming a run is done.

Recurring nightly run path:
- launchd label: `<launchd_label>`
- schedule: daily `01:00` local host time
- installer command: `bash scripts/install_nightly_postcheck_launchd.sh`

## Known gotchas
- `--all` is still accepted but deprecated in favor of `--pipeline full`.
- Replay + calibration phases can appear low-CPU while still making DB progress.
- `ML_FEATURE_CONTRACT` is strict; feature order drift requires retraining and artifact refresh.
- Archived planning docs exist under `old/`; do not treat them as current runbooks.

## Documentation map
Start here:
- `README.md`: quick start and command index.
- `overview.md` (this file): architecture + internals + operations reference.

Operational memory docs:
- `handoff.md`: active execution state and next actions.
- `changes.md`: chronological change log.
- `todo.md`: actionable backlog.
- `dontdo.md`: known failures/pitfalls.

Historical/archived:
- `old/` and `docs-archive/`.
