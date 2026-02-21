#!/bin/bash

# Stop Market Analyzer Dashboard
# This script cleanly stops the dashboard process

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${MARKET_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PID_FILE="$PROJECT_ROOT/dashboard.pid"

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "Dashboard is not running (no PID file found)"
    
    # Try to find and kill any running dashboard.py processes
    PIDS=$(pgrep -f "python.*dashboard.py" || true)
    if [ -n "$PIDS" ]; then
        echo "Found running dashboard processes: $PIDS"
        echo "Stopping them..."
        kill $PIDS 2>/dev/null || true
        sleep 1
        # Force kill if still running
        kill -9 $PIDS 2>/dev/null || true
        echo "Dashboard processes stopped"
    fi
    
    exit 0
fi

# Read PID
PID=$(cat "$PID_FILE")

# Check if process is running
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "Dashboard process (PID: $PID) is not running"
    rm -f "$PID_FILE"
    exit 0
fi

# Stop the process gracefully
echo "Stopping dashboard (PID: $PID)..."
kill "$PID" 2>/dev/null || true

# Wait up to 5 seconds for graceful shutdown
for i in {1..5}; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo "Dashboard stopped successfully"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
if ps -p "$PID" > /dev/null 2>&1; then
    echo "Forcing shutdown..."
    kill -9 "$PID" 2>/dev/null || true
    sleep 1
fi

# Clean up PID file
rm -f "$PID_FILE"
echo "Dashboard stopped"
exit 0
