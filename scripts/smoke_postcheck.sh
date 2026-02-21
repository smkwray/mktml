#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--project-root PATH] [--integration-start DATE] [--integration-end DATE]

Run post-check orchestration in a deterministic local smoke mode without triggering full
pipeline workloads or remote scheduling.

Options:
  --project-root PATH           Project root (default: script directory parent)
  --integration-start DATE      Replay integration start date (YYYY-MM-DD)
  --integration-end DATE        Replay integration end date (YYYY-MM-DD)
  --help                        Show this help text
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

DEFAULT_SMOKE_DATE="$(
python3 - <<'PY'
import datetime
print((datetime.date.today() - datetime.timedelta(days=1)).isoformat())
PY
)"
INTEGRATION_START="${MARKET_POSTCHECK_SMOKE_INTEGRATION_START:-$DEFAULT_SMOKE_DATE}"
INTEGRATION_END="${MARKET_POSTCHECK_SMOKE_INTEGRATION_END:-$DEFAULT_SMOKE_DATE}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root)
      PROJECT_ROOT="$2"
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
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/market_postcheck_smoke_XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

FULL_RUN_LOG="$TMP_DIR/full_run.log"
POSTCHECK_LOG="$TMP_DIR/postcheck.log"
FULL_RUN_PID_FILE="$TMP_DIR/fake_full_run.pid"
FAKE_PYTHON="$TMP_DIR/fake_postcheck_python.sh"
REAL_PYTHON="$(command -v python3)"

cat <<'EOF' > "$FULL_RUN_LOG"
END full pipeline rc=0
EOF
printf '999999' > "$FULL_RUN_PID_FILE"

cat <<'EOF' > "$FAKE_PYTHON"
#!/usr/bin/env bash
set -euo pipefail

real_python="${MARKET_POSTCHECK_SMOKE_REAL_PYTHON:-$(command -v python3)}"

if (( $# == 0 )); then
  exit 0
fi

case "$1" in
  *validate_replay_calibration_integration.py)
    exit 0
    ;;
  *check_model_runtime_skew.py)
    exit 0
    ;;
  *)
    exec "$real_python" "$@"
    ;;
esac
EOF
chmod +x "$FAKE_PYTHON"

postcheck_rc=0
if ! bash "$PROJECT_ROOT/scripts/postcheck_after_full_run.sh" \
  --project-root "$PROJECT_ROOT" \
  --full-run-log "$FULL_RUN_LOG" \
  --postcheck-log "$POSTCHECK_LOG" \
  --full-run-pid-file "$FULL_RUN_PID_FILE" \
  --integration-start "$INTEGRATION_START" \
  --integration-end "$INTEGRATION_END" \
  --integration-replays 1 \
  --integration-flush-rows 5 \
  --python-bin "$FAKE_PYTHON" \
  --runtime-model-dir "$PROJECT_ROOT/models" \
  --full-run-flush-seconds 0 \
  --no-runtime-strict; then
  postcheck_rc=$?
fi

missing_marker=0
for marker in \
  "full_run_rc=" \
  "integration_rc=" \
  "runtime_skew_rc=" \
  "END post-check rc="
do
  if ! grep -Fq -- "$marker" "$POSTCHECK_LOG"; then
    echo "ERROR: marker not found in ${POSTCHECK_LOG}: $marker" >&2
    missing_marker=1
  else
    grep -Fm1 -- "$marker" "$POSTCHECK_LOG"
  fi
done

if [[ "$postcheck_rc" -ne 0 ]]; then
  echo "Smoke post-check command failed with rc=${postcheck_rc}" >&2
  exit "$postcheck_rc"
fi

if (( missing_marker != 0 )); then
  echo "Smoke marker validation failed" >&2
  exit 1
fi

echo "Smoke marker validation passed"
exit 0
