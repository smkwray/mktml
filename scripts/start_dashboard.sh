#!/bin/bash

# Start Market Analyzer Dashboard
# This script starts the dashboard with proper environment setup and logging

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${MARKET_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_PATH="${MARKET_VENV_PATH:-${HOME}/venvs/market_analyzer}"
DASHBOARD_SCRIPT="$PROJECT_ROOT/src/dashboard.py"
PID_FILE="$PROJECT_ROOT/dashboard.pid"
LOG_DIR="$HOME/Library/Logs"
LOG_FILE="$LOG_DIR/market-dashboard.log"
ERROR_LOG="$LOG_DIR/market-dashboard.error.log"
SSH_TUNNEL_TARGET="${MARKET_DASHBOARD_SSH_TARGET:-<user>@<remote-host>}"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Dashboard is already running (PID: $PID)"
        exit 0
    else
        echo "Removing stale PID file"
        rm -f "$PID_FILE"
    fi
fi

# Change to project directory
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH" >&2
    echo "Please create it with: python3 -m venv $VENV_PATH" >&2
    exit 1
fi

# Check if dashboard.py exists
if [ ! -f "$DASHBOARD_SCRIPT" ]; then
    echo "ERROR: Dashboard script not found at $DASHBOARD_SCRIPT" >&2
    exit 1
fi

# Start dashboard in background
echo "Starting Market Analyzer Dashboard..."
echo "Logs: $LOG_FILE"
echo "Errors: $ERROR_LOG"

nohup python3 "$DASHBOARD_SCRIPT" >> "$LOG_FILE" 2>> "$ERROR_LOG" &
DASHBOARD_PID=$!

# Save PID
echo $DASHBOARD_PID > "$PID_FILE"

# Wait a moment to ensure it started successfully
sleep 2

# Check if process is still running
if ps -p "$DASHBOARD_PID" > /dev/null 2>&1; then
    echo "Dashboard started successfully (PID: $DASHBOARD_PID)"
    echo "Access locally at: http://127.0.0.1:5050"
    echo "Access remotely via SSH tunnel:"
    echo "  ssh -L 5050:127.0.0.1:5050 ${SSH_TUNNEL_TARGET}"
    echo "  Then open: http://localhost:5050"
    exit 0
else
    echo "ERROR: Dashboard failed to start. Check logs at $ERROR_LOG" >&2
    rm -f "$PID_FILE"
    exit 1
fi
