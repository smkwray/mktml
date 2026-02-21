#!/usr/bin/env bash
set -euo pipefail

SCRIPT_VERSION="1"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Post-run orchestration wrapper that waits for a full-run process completion,
runs integration and runtime checks, then emits RC markers for dashboard parsing.

Options:
  --project-root PATH            Project root (default: script parent, env MARKET_POSTCHECK_PROJECT_ROOT)
  --full-run-log PATH            Full-run log to tail/parse (default: logs/full_run_<timestamp>.log)
  --postcheck-log PATH           Post-check log path (default: logs/postcheck_<timestamp>.log)
  --full-run-pid PID             Full-run PID to wait on
  --full-run-pid-file PATH       File containing full-run PID
  --full-run-pattern PATTERN      Pattern to locate full-run process (fallback)
  --full-run-cmd CMD             Run this command and wait for exit; sets/uses full-run log
  --full-run-timeout-seconds N    Max seconds to wait for process (default 86400)
  --full-run-wait-interval N      Poll interval seconds (default 60)
  --full-run-flush-seconds N      Sleep after process exit to allow log flush (default 30)
  --integration-start DATE        Replay calibration date window start (YYYY-MM-DD)
  --integration-end DATE          Replay calibration date window end (YYYY-MM-DD)
  --integration-replays N         Integration replay count (default 2)
  --integration-flush-rows N      Optional flush-rows override for integration gate
  --runtime-model-dir PATH        Model dir for runtime skew gate (default <project>/models)
  --runtime-strict / --no-runtime-strict  Runtime gate strict mode (default strict)
  --python-bin PATH               Python executable (default: env POSTCHECK_PYTHON)
  --help                         Show this help text

Environment variables:
  MARKET_POSTCHECK_PROJECT_ROOT
  MARKET_POSTCHECK_FULL_RUN_LOG
  MARKET_POSTCHECK_POSTCHECK_LOG
  MARKET_POSTCHECK_FULL_RUN_PID
  MARKET_POSTCHECK_FULL_RUN_PID_FILE
  MARKET_POSTCHECK_FULL_RUN_PATTERN
  MARKET_POSTCHECK_FULL_RUN_CMD
  MARKET_POSTCHECK_FULL_RUN_TIMEOUT_SECONDS
  MARKET_POSTCHECK_FULL_RUN_WAIT_INTERVAL_SECONDS
  MARKET_POSTCHECK_FULL_RUN_FLUSH_SECONDS
  MARKET_POSTCHECK_INTEGRATION_START
  MARKET_POSTCHECK_INTEGRATION_END
  MARKET_POSTCHECK_INTEGRATION_REPLAYS
  MARKET_POSTCHECK_INTEGRATION_FLUSH_ROWS
  MARKET_POSTCHECK_RUNTIME_MODEL_DIR
  MARKET_POSTCHECK_RUNTIME_STRICT
  MARKET_POSTCHECK_PYTHON

RC lines emitted:
  full_run_rc=<int>
  integration_rc=<int>
  runtime_skew_rc=<int>
  END post-check rc=<int>
USAGE
  exit 0
}

safe_timestamp() {
  date '+%Y%m%d_%H%M%S'
}

safe_now() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

default_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_ts="$(safe_timestamp)"
default_start="$(
python3 - <<'PY'
import datetime
print((datetime.date.today() - datetime.timedelta(days=1)).isoformat())
PY
)"

SCRIPT_TS="${MARKET_POSTCHECK_TIMESTAMP:-$default_ts}"
PROJECT_ROOT="${MARKET_POSTCHECK_PROJECT_ROOT:-$default_root}"
POSTCHECK_PYTHON="${MARKET_POSTCHECK_PYTHON:-python3}"
FULL_RUN_LOG="${MARKET_POSTCHECK_FULL_RUN_LOG:-}"
POSTCHECK_LOG="${MARKET_POSTCHECK_POSTCHECK_LOG:-}"
FULL_RUN_PID="${MARKET_POSTCHECK_FULL_RUN_PID:-}"
FULL_RUN_PID_FILE="${MARKET_POSTCHECK_FULL_RUN_PID_FILE:-}"
FULL_RUN_PATTERN="${MARKET_POSTCHECK_FULL_RUN_PATTERN:-$PROJECT_ROOT/src/main.py --all}"
FULL_RUN_CMD="${MARKET_POSTCHECK_FULL_RUN_CMD:-}"
FULL_RUN_TIMEOUT_SECONDS="${MARKET_POSTCHECK_FULL_RUN_TIMEOUT_SECONDS:-86400}"
FULL_RUN_WAIT_INTERVAL_SECONDS="${MARKET_POSTCHECK_FULL_RUN_WAIT_INTERVAL_SECONDS:-60}"
FULL_RUN_FLUSH_SECONDS="${MARKET_POSTCHECK_FULL_RUN_FLUSH_SECONDS:-30}"
INTEGRATION_START="${MARKET_POSTCHECK_INTEGRATION_START:-$default_start}"
INTEGRATION_END="${MARKET_POSTCHECK_INTEGRATION_END:-$default_start}"
INTEGRATION_REPLAYS="${MARKET_POSTCHECK_INTEGRATION_REPLAYS:-2}"
INTEGRATION_FLUSH_ROWS="${MARKET_POSTCHECK_INTEGRATION_FLUSH_ROWS:-}"
RUNTIME_MODEL_DIR="${MARKET_POSTCHECK_RUNTIME_MODEL_DIR:-$PROJECT_ROOT/models}"
RUNTIME_STRICT="${MARKET_POSTCHECK_RUNTIME_STRICT:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root)
      PROJECT_ROOT="$2"
      shift 2
      ;;
    --full-run-log)
      FULL_RUN_LOG="$2"
      shift 2
      ;;
    --postcheck-log)
      POSTCHECK_LOG="$2"
      shift 2
      ;;
    --full-run-pid)
      FULL_RUN_PID="$2"
      shift 2
      ;;
    --full-run-pid-file)
      FULL_RUN_PID_FILE="$2"
      shift 2
      ;;
    --full-run-pattern)
      FULL_RUN_PATTERN="$2"
      shift 2
      ;;
    --full-run-cmd)
      FULL_RUN_CMD="$2"
      shift 2
      ;;
    --full-run-timeout-seconds)
      FULL_RUN_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --full-run-wait-interval)
      FULL_RUN_WAIT_INTERVAL_SECONDS="$2"
      shift 2
      ;;
    --full-run-flush-seconds)
      FULL_RUN_FLUSH_SECONDS="$2"
      shift 2
      ;;
    --integration-start)
      INTEGRATION_START="$2"
      shift 2
      ;;
    --integration-end)
      INTEGRATION_END="$2"
      shift 2
      ;;
    --integration-replays)
      INTEGRATION_REPLAYS="$2"
      shift 2
      ;;
    --integration-flush-rows)
      INTEGRATION_FLUSH_ROWS="$2"
      shift 2
      ;;
    --runtime-model-dir)
      RUNTIME_MODEL_DIR="$2"
      shift 2
      ;;
    --runtime-strict)
      RUNTIME_STRICT=1
      shift
      ;;
    --no-runtime-strict)
      RUNTIME_STRICT=0
      shift
      ;;
    --python-bin)
      POSTCHECK_PYTHON="$2"
      shift 2
      ;;
    --help|-h)
      usage
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      ;;
  esac
done

sanitize_integer() {
  local value="$1"
  local label="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "Invalid integer for ${label}: ${value}" >&2
    exit 2
  fi
}

validate_iso_date() {
  local value="$1"
  local label="$2"
  if [[ ! "$value" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "Invalid date for ${label}: ${value} (expected YYYY-MM-DD)" >&2
    exit 2
  fi
}

sanitize_integer "$FULL_RUN_TIMEOUT_SECONDS" "--full-run-timeout-seconds"
sanitize_integer "$FULL_RUN_WAIT_INTERVAL_SECONDS" "--full-run-wait-interval"
sanitize_integer "$FULL_RUN_FLUSH_SECONDS" "--full-run-flush-seconds"
sanitize_integer "$INTEGRATION_REPLAYS" "--integration-replays"
validate_iso_date "$INTEGRATION_START" "--integration-start"
validate_iso_date "$INTEGRATION_END" "--integration-end"
if [[ "$INTEGRATION_START" > "$INTEGRATION_END" ]]; then
  echo "Invalid date window: start > end (${INTEGRATION_START} > ${INTEGRATION_END})" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
FULL_RUN_LOG="${FULL_RUN_LOG:-$LOG_DIR/full_run_${SCRIPT_TS}.log}"
POSTCHECK_LOG="${POSTCHECK_LOG:-$LOG_DIR/postcheck_${SCRIPT_TS}.log}"
mkdir -p "$(dirname "$POSTCHECK_LOG")"
mkdir -p "$(dirname "$FULL_RUN_LOG")"

log() {
  local message="$1"
  printf '[%s] %s\n' "$(safe_now)" "$message" | tee -a "$POSTCHECK_LOG" >&2
}

extract_marker_rc() {
  local logfile="$1"
  local marker="${2:-END full pipeline rc=}"
  if [[ ! -f "$logfile" ]]; then
    echo ""
    return
  fi
  tail -n 400 "$logfile" \
    | grep -E "${marker}[0-9]+" \
    | tail -n 1 \
    | sed -E 's/.*rc=([0-9]+).*/\1/'
}

find_pid_from_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo ""
    return
  fi
  tr -d '[:space:]' < "$path"
}

find_pid_by_pattern() {
  local pattern="$1"
  if [[ -z "$pattern" ]]; then
    echo ""
    return
  fi
  pgrep -f -- "$pattern" | head -n 1 || true
}

wait_for_process() {
  local pid="$1"
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if (( waited >= FULL_RUN_TIMEOUT_SECONDS )); then
      return 124
    fi
    local minutes=$(( waited / 60 ))
    local timeout_minutes=$(( FULL_RUN_TIMEOUT_SECONDS / 60 ))
    log "full run active; sleep ${FULL_RUN_WAIT_INTERVAL_SECONDS}s (waited ${minutes}m/${timeout_minutes}m)"
    sleep "$FULL_RUN_WAIT_INTERVAL_SECONDS"
    waited=$(( waited + FULL_RUN_WAIT_INTERVAL_SECONDS ))
  done
  log "full run process exited; allowing log flush (${FULL_RUN_FLUSH_SECONDS}s)"
  sleep "$FULL_RUN_FLUSH_SECONDS"
  return 0
}

run_full_pipeline_command() {
  local rc=0
  local full_dir
  full_dir="$(dirname "$FULL_RUN_LOG")"
  mkdir -p "$full_dir"
  log "Starting full-run command: ${FULL_RUN_CMD}"
  set +e
  ( cd "$PROJECT_ROOT" && bash -c "$FULL_RUN_CMD" ) >> "$FULL_RUN_LOG" 2>&1
  rc=$?
  set -e
  echo "$rc"
}

run_integration_gate() {
  local cmd=(
    "$POSTCHECK_PYTHON"
    "$PROJECT_ROOT/scripts/validate_replay_calibration_integration.py"
    "--start" "$INTEGRATION_START"
    "--end" "$INTEGRATION_END"
    "--replays" "$INTEGRATION_REPLAYS"
  )
  if [[ -n "$INTEGRATION_FLUSH_ROWS" ]]; then
    cmd+=("--flush-rows" "$INTEGRATION_FLUSH_ROWS")
  fi
  local rc=0
  log "Running integration gate (${INTEGRATION_START}..${INTEGRATION_END})"
  set +e
  "${cmd[@]}"
  rc=$?
  set -e
  echo "$rc"
}

run_runtime_skew_gate() {
  local cmd=(
    "$POSTCHECK_PYTHON"
    "$PROJECT_ROOT/scripts/check_model_runtime_skew.py"
    "--model-dir" "$RUNTIME_MODEL_DIR"
  )
  local rc=0
  if [[ "$RUNTIME_STRICT" == "1" || "$RUNTIME_STRICT" == "true" ]]; then
    cmd+=("--strict")
  fi
  log "Running runtime skew strict=${RUNTIME_STRICT}"
  set +e
  "${cmd[@]}"
  rc=$?
  set -e
  echo "$rc"
}

log "START post-check wrapper (version=$SCRIPT_VERSION)"
log "Project root: $PROJECT_ROOT"
log "Full run log: $FULL_RUN_LOG"
log "Post-check log: $POSTCHECK_LOG"

full_run_rc=2
if [[ -n "$FULL_RUN_CMD" ]]; then
  full_run_rc="$(run_full_pipeline_command)"
elif [[ -n "$FULL_RUN_PID" ]]; then
  if kill -0 "$FULL_RUN_PID" 2>/dev/null; then
    if ! wait_for_process "$FULL_RUN_PID"; then
      full_run_rc=124
    else
      full_run_rc="$(extract_marker_rc "$FULL_RUN_LOG" 'END full pipeline rc=')"
      if [[ -z "$full_run_rc" ]]; then
        full_run_rc=2
      fi
    fi
  else
    full_run_rc="$(extract_marker_rc "$FULL_RUN_LOG" 'END full pipeline rc=')"
    if [[ -z "$full_run_rc" ]]; then
      full_run_rc=2
    fi
  fi
else
  if [[ -n "$FULL_RUN_PID_FILE" && -f "$FULL_RUN_PID_FILE" ]]; then
    FULL_RUN_PID="$(find_pid_from_file "$FULL_RUN_PID_FILE")"
  fi
  if [[ -z "${FULL_RUN_PID:-}" && -n "$FULL_RUN_PATTERN" ]]; then
    FULL_RUN_PID="$(find_pid_by_pattern "$FULL_RUN_PATTERN")"
  fi

  if [[ -n "$FULL_RUN_PID" ]]; then
    if ! wait_for_process "$FULL_RUN_PID"; then
      full_run_rc=124
    else
      full_run_rc="$(extract_marker_rc "$FULL_RUN_LOG" 'END full pipeline rc=')"
      if [[ -z "$full_run_rc" ]]; then
        full_run_rc=2
      fi
    fi
  else
    full_run_rc="$(extract_marker_rc "$FULL_RUN_LOG" 'END full pipeline rc=')"
    if [[ -z "$full_run_rc" ]]; then
      full_run_rc=2
    fi
  fi
fi

log "full_run_rc=${full_run_rc}"

integration_rc=0
runtime_skew_rc=0
if [[ "$full_run_rc" -eq 0 ]]; then
  integration_rc="$(run_integration_gate)"
else
  integration_rc=2
fi
log "integration_rc=${integration_rc}"

if [[ "$integration_rc" -eq 0 ]]; then
  runtime_skew_rc="$(run_runtime_skew_gate)"
else
  runtime_skew_rc=2
fi
log "runtime_skew_rc=${runtime_skew_rc}"

postcheck_rc=0
if [[ "$full_run_rc" -ne 0 || "$integration_rc" -ne 0 || "$runtime_skew_rc" -ne 0 ]]; then
  postcheck_rc=1
fi

log "END post-check rc=${postcheck_rc}"
exit "$postcheck_rc"
