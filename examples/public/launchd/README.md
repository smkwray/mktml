# launchd Templates (Public)

These templates are safe-to-share launchd examples for macOS.

## Files
- `com.market-nightly-postcheck.plist`
- `com.market-dashboard.plist`

## Placeholder replacement
Replace these strings before install:
- `__PROJECT_ROOT__` -> absolute project path
- `__HOME__` -> deployment account home path
- `__PYTHON_BIN__` -> python executable for this project
- `__VENV_PATH__` -> virtualenv path
- `__SSH_TUNNEL_TARGET__` -> SSH target for remote dashboard access (`user@host`)

## Install example
```bash
cp com.market-nightly-postcheck.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.market-nightly-postcheck.plist >/dev/null 2>&1 || true
launchctl load ~/Library/LaunchAgents/com.market-nightly-postcheck.plist
launchctl list | grep com.market-nightly-postcheck
```
