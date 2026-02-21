#!/usr/bin/env python3
"""
Update Safe Asset Yields via Gemini CLI
========================================
Generates a prompt for gemini-cli to research current yields for safe assets,
then parses the response and updates data/safe_asset_yields.json.

Usage (on remote device with gemini-cli installed):
    python scripts/update_yields.py

The script will:
1. Read SAFE_ASSET_ALLOWLIST from config
2. Generate a structured prompt for gemini-cli
3. Run gemini-cli with the prompt
4. Parse JSON response and save to data/safe_asset_yields.json
"""

import os
import sys
import json
import subprocess
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import SAFE_ASSET_ALLOWLIST, SAFE_ASSET_YIELDS_FILE

PRIMARY_MODEL = "gemini-3-flash-preview"  # Use flash as primary (faster, less rate-limited)
FALLBACK_MODEL = None  # None means use default gemini-cli (no -m flag)


def generate_prompt(tickers: list) -> str:
    """Generate a structured prompt for gemini-cli to research yields."""
    ticker_list = ", ".join(tickers)
    return f"""You are a financial data assistant. Return ONLY valid JSON with no additional text, markdown, or commentary.

Research the current SEC 30-day yield or distribution yield for these ETFs/funds: {ticker_list}

Return data in this exact JSON format:
{{"yields": {{"TICKER1": 0.045, "TICKER2": 0.038, ...}}}}

Rules:
- Use decimal format (e.g., 4.5% = 0.045)
- If yield is unknown, use 0.0
- No commentary, no markdown, no code blocks - ONLY the JSON object
- Focus on SEC yield for bond funds, distribution yield for others"""


def run_gemini_cli(prompt: str, model: str = PRIMARY_MODEL) -> tuple:
    """Run gemini-cli with the given prompt. Returns (output, error_type).
    
    Args:
        prompt: The prompt to send to Gemini
        model: Model to use. Defaults to PRIMARY_MODEL (flash). Pass None for default gemini.
    
    Returns:
        (output_string, error_type) where error_type is None on success,
        'rate_limit' on rate limiting, 'daily_quota' on quota exhaustion,
        or other error strings.
    """
    import re
    
    # Check multiple possible locations
    gemini_paths = [
        "gemini",  # PATH
        "gemini-cli",  # Alternative name
        "/opt/homebrew/bin/gemini",  # macOS Homebrew
        "/usr/local/bin/gemini",  # Linux/macOS
        os.path.expanduser("~/.local/bin/gemini"),  # User local
    ]
    
    for gemini_cmd in gemini_paths:
        try:
            cmd = [gemini_cmd]
            if model:
                cmd.extend(["-m", model])
            cmd.extend(["-p", prompt])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min for long multi-ticker prompts
            )
            
            combined = (result.stdout + result.stderr)
            combined_lower = combined.lower()
            
            # 1. Hard Quota Detection (Daily limit / Resource Exhausted)
            hard_quota_keywords = [
                'quota exceeded', 'quota limit', 'insufficient quota', 
                'resource exhausted'
            ]
            if any(k in combined_lower for k in hard_quota_keywords):
                print(f"[update_yields] Detected hard quota limit")
                return combined, "daily_quota"

            # 2. Soft Rate Limit Detection (429 / Too Many Requests)
            rate_limit_keywords = [
                'too many requests', '429 too many requests'
            ]
            if any(k in combined_lower for k in rate_limit_keywords):
                print(f"[update_yields] Detected rate limit")
                return combined, "rate_limit"

            # 3. HTTP 429 Detection (Contextual)
            if '429' in combined:
                if 'error' in combined_lower or 'status' in combined_lower or 'code' in combined_lower:
                    print(f"[update_yields] Detected '429' error")
                    return combined, "rate_limit"
            
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip(), None
            
            # If we got here, it might be a silent failure
            if result.stderr:
                return result.stderr.strip(), "execution_error"
                 
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            print("[update_yields] ERROR: gemini-cli timed out after 300s.")
            return "", "timeout"
        except Exception as e:
            print(f"[update_yields] ERROR running {gemini_cmd}: {e}")
            continue
    
    print("[update_yields] ERROR: Could not find gemini-cli in any known location.")
    return "", "not_found"


def parse_response(response: str) -> dict:
    """Parse gemini-cli JSON response, handling common formatting issues."""
    if not response:
        return {}
    
    # Strip markdown code blocks if present
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        # Remove first and last lines (code block markers)
        lines = [l for l in lines if not l.strip().startswith("```")]
        response = "\n".join(lines)
    
    try:
        data = json.loads(response)
        if "yields" in data:
            # Validate and filter yields
            yields = {}
            for ticker, value in data["yields"].items():
                try:
                    val = float(value)
                    if 0 <= val <= 1:  # Reasonable yield range
                        yields[ticker.upper()] = val
                except (TypeError, ValueError):
                    pass
            return yields
    except json.JSONDecodeError as e:
        print(f"[update_yields] JSON parse error: {e}")
        print(f"[update_yields] Raw response: {response[:500]}...")
    
    return {}


def update_yields_file(yields: dict, output_path: str) -> bool:
    """Update the yields JSON file with new data."""
    try:
        data = {
            "_meta": {
                "description": "Safe asset yield data. Auto-populated by scripts/update_yields.py via gemini-cli.",
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker_count": len(yields),
            },
            "yields": yields,
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"[update_yields] SUCCESS: Saved {len(yields)} yields to {output_path}")
        return True
    except Exception as e:
        print(f"[update_yields] ERROR writing file: {e}")
        return False


def main():
    print(f"[update_yields] Researching yields for {len(SAFE_ASSET_ALLOWLIST)} tickers...")
    
    prompt = generate_prompt(SAFE_ASSET_ALLOWLIST)
    print(f"[update_yields] Running gemini-cli with primary model ({PRIMARY_MODEL})...")
    
    response, error = run_gemini_cli(prompt)  # Uses PRIMARY_MODEL by default
    
    # Handle rate limit with fallback model (default gemini, no -m flag)
    if error == "rate_limit":
        print(f"[update_yields] Rate limit hit on primary. Trying fallback model (default gemini)...")
        response, error = run_gemini_cli(prompt, model=FALLBACK_MODEL)  # None = default gemini
        if error == "rate_limit":
            print(f"[update_yields] Fallback model (default gemini) also rate limited. Exiting.")
            sys.exit(1)
    
    # Handle hard quota
    if error == "daily_quota":
        print("[update_yields] Daily quota exhausted. Exiting.")
        sys.exit(1)
    
    if not response:
        print(f"[update_yields] No response from gemini-cli (error: {error}). Exiting.")
        sys.exit(1)
    
    yields = parse_response(response)
    if not yields:
        print("[update_yields] Failed to parse yields. Exiting.")
        sys.exit(1)
    
    success = update_yields_file(yields, SAFE_ASSET_YIELDS_FILE)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

