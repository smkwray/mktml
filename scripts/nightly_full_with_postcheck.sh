#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${MARKET_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${MARKET_PYTHON_BIN:-${HOME}/venvs/market_analyzer/bin/python}"
POSTCHECK_WRAPPER="${PROJECT_ROOT}/scripts/postcheck_after_full_run.sh"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python binary not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -f "${POSTCHECK_WRAPPER}" ]]; then
  echo "ERROR: Missing wrapper script: ${POSTCHECK_WRAPPER}" >&2
  exit 2
fi

TS="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${PROJECT_ROOT}/logs"
FULL_LOG="${LOG_DIR}/full_run_${TS}.log"
POST_LOG="${LOG_DIR}/postcheck_${TS}.log"

mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/reports"

FULL_CMD="PYTHONDONTWRITEBYTECODE=1 ${PYTHON_BIN} src/main.py --pipeline full"

# Reserve a 48h wait window in case training/replay are unusually long.
TIMEOUT_SECONDS="${MARKET_POSTCHECK_FULL_RUN_TIMEOUT_SECONDS:-172800}"

exec /bin/bash "${POSTCHECK_WRAPPER}" \
  --project-root "${PROJECT_ROOT}" \
  --python-bin "${PYTHON_BIN}" \
  --full-run-log "${FULL_LOG}" \
  --postcheck-log "${POST_LOG}" \
  --full-run-cmd "${FULL_CMD}" \
  --full-run-timeout-seconds "${TIMEOUT_SECONDS}" \
  --integration-replays "${MARKET_POSTCHECK_INTEGRATION_REPLAYS:-2}" \
  --runtime-model-dir "${PROJECT_ROOT}/models"
