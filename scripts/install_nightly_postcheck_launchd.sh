#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${MARKET_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PLIST_SRC="${MARKET_NIGHTLY_PLIST_SRC:-}"

if [[ -z "${PLIST_SRC}" ]]; then
  if [[ -f "${PROJECT_ROOT}/scripts/com.market-nightly-postcheck.plist" ]]; then
    PLIST_SRC="${PROJECT_ROOT}/scripts/com.market-nightly-postcheck.plist"
  else
    shopt -s nullglob
    candidates=("${PROJECT_ROOT}"/scripts/com.*market-nightly-postcheck.plist)
    shopt -u nullglob
    if [[ "${#candidates[@]}" -gt 0 ]]; then
      PLIST_SRC="${candidates[0]}"
    fi
  fi
fi

if [[ -z "${PLIST_SRC}" || ! -f "${PLIST_SRC}" ]]; then
  echo "ERROR: Missing nightly plist. Set MARKET_NIGHTLY_PLIST_SRC or add scripts/com.market-nightly-postcheck.plist." >&2
  exit 2
fi

LABEL="${MARKET_NIGHTLY_LABEL:-$(basename "${PLIST_SRC}" .plist)}"
PLIST_DST="${HOME}/Library/LaunchAgents/${LABEL}.plist"


mkdir -p "${HOME}/Library/LaunchAgents"
mkdir -p "${HOME}/Library/Logs"

cp "${PLIST_SRC}" "${PLIST_DST}"
plutil -lint "${PLIST_DST}" >/dev/null

if launchctl list | grep -q "${LABEL}"; then
  launchctl unload "${PLIST_DST}" >/dev/null 2>&1 || true
fi

launchctl load "${PLIST_DST}"

echo "Installed and loaded: ${LABEL}"
launchctl list | grep "${LABEL}" || true
