#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[public-guard] Not inside a git work tree; skipping." >&2
  exit 0
fi

GIT_ROOT="$(git rev-parse --show-toplevel)"
cd "$GIT_ROOT"

if [[ "$PROJECT_ROOT" == "$GIT_ROOT" ]]; then
  SCOPE_PATHSPEC="."
else
  if [[ "${PUBLIC_GUARD_ALLOW_SUBDIR_REPO:-0}" != "1" ]]; then
    echo "[public-guard] BLOCKED: git root is not the project root." >&2
    echo "[public-guard] git root:    $GIT_ROOT" >&2
    echo "[public-guard] project root: $PROJECT_ROOT" >&2
    echo "[public-guard] Create/use a dedicated repo at the project root before public push." >&2
    echo "[public-guard] (Override only if intentional: PUBLIC_GUARD_ALLOW_SUBDIR_REPO=1)" >&2
    exit 1
  fi
  SCOPE_PATHSPEC="${PROJECT_ROOT#$GIT_ROOT/}"
fi

in_scope() {
  local f="$1"
  if [[ "$SCOPE_PATHSPEC" == "." ]]; then
    return 0
  fi
  [[ "$f" == "$SCOPE_PATHSPEC" || "$f" == "$SCOPE_PATHSPEC/"* ]]
}

to_local_path() {
  local f="$1"
  if [[ "$SCOPE_PATHSPEC" == "." ]]; then
    printf '%s\n' "$f"
  else
    printf '%s\n' "${f#$SCOPE_PATHSPEC/}"
  fi
}

# Determine file scope: outgoing commit range if upstream exists, else tracked files.
FILES=()
upstream=""
if upstream=$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null); then
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    in_scope "$line" || continue
    FILES+=("$line")
  done < <(git diff --name-only "${upstream}"..HEAD)
fi

if [[ "${#FILES[@]}" -eq 0 ]]; then
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    in_scope "$line" || continue
    FILES+=("$line")
  done < <(git ls-files -- "$SCOPE_PATHSPEC")
fi

# First-push fallback: scan non-ignored files in the project subtree.
if [[ "${#FILES[@]}" -eq 0 ]]; then
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    in_scope "$line" || continue
    FILES+=("$line")
  done < <(git ls-files -co --exclude-standard -- "$SCOPE_PATHSPEC")
fi

if [[ "${#FILES[@]}" -eq 0 ]]; then
  echo "[public-guard] No files to scan." >&2
  exit 0
fi

SCAN_FILES=()
LOCAL_SCAN_FILES=()
for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || continue
  local_f="$(to_local_path "$f")"
  SCAN_FILES+=("$f")
  LOCAL_SCAN_FILES+=("$local_f")
done

if [[ "${#SCAN_FILES[@]}" -eq 0 ]]; then
  echo "[public-guard] No existing files to scan." >&2
  exit 0
fi

blocked_path_patterns=(
  '^config\\.py$'
  '^data/'
  '^models/'
  '^reports/'
  '^reports-archive/'
  '^logs/'
  '^old/'
  '^docs-archive/'
  '^.*-archive/'
  '^SCAN_STATUS\\.md$'
  '^TRAIN_STATUS\\.md$'
  '^AGENTS\\.md$'
  '^CLAUDE\\.md$'
  '^plan\\.md$'
  '^qual_metrics_plan\\.md$'
  '^handoff\\.md$'
  '^changes\\.md$'
  '^todo\\.md$'
  '^prompt\\.md$'
  '^dontdo\\.md$'
  '^memory-system\\.md$'
  '^spark-template\\.md$'
  '^spark[0-9]+\\.md$'
  '^run_sparks\\.sh$'
  '^prompt/'
  '^scripts/com\\..*\\.plist$'
)

blocked_content_patterns=(
  '/Users/[A-Za-z0-9._-]+/'
  'ssh[[:space:]]+[A-Za-z0-9._-]+@([0-9]{1,3}\\.){3}[0-9]{1,3}'
  'ALPACA_(API|SECRET)_KEY[[:space:]]*=[[:space:]]*os\\.environ\\.get\\([^,]+,[[:space:]]*\x27[^\x27]{8,}\x27\\)'
  'FRED_API_KEY[[:space:]]*=[[:space:]]*\x22[A-Za-z0-9]{12,}\x22'
  '(API_KEY|SECRET_KEY|TOKEN|PASSWORD)[[:space:]]*[:=][[:space:]]*[\x22\x27][A-Za-z0-9_\\-]{20,}[\x22\x27]'
)

path_hits=()
for local_f in "${LOCAL_SCAN_FILES[@]}"; do
  for pat in "${blocked_path_patterns[@]}"; do
    if [[ "$local_f" =~ $pat ]]; then
      path_hits+=("$local_f")
      break
    fi
  done
done

rg_args=()
for pat in "${blocked_content_patterns[@]}"; do
  rg_args+=("-e" "$pat")
done
rg_args+=("--")
for f in "${SCAN_FILES[@]}"; do
  rg_args+=("$f")
done

content_hits=()
while IFS= read -r line; do
  [[ -n "$line" ]] && content_hits+=("$line")
done < <(rg -n -I --no-heading --color never "${rg_args[@]}" 2>/dev/null || true)

if [[ "${#path_hits[@]}" -gt 0 || "${#content_hits[@]}" -gt 0 ]]; then
  echo "[public-guard] BLOCKED: possible private/sensitive content detected." >&2

  if [[ "${#path_hits[@]}" -gt 0 ]]; then
    echo "[public-guard] Blocked paths:" >&2
    printf '  - %s\n' "${path_hits[@]}" >&2
  fi

  if [[ "${#content_hits[@]}" -gt 0 ]]; then
    echo "[public-guard] Content matches:" >&2
    printf '  - %s\n' "${content_hits[@]}" >&2
  fi

  echo "[public-guard] Fix/redact or unstage files before pushing." >&2
  exit 1
fi

echo "[public-guard] OK: no blocked paths/content in push scope." >&2
exit 0
