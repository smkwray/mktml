
import subprocess
import sys

def run_debug():
    prompt = "Tell me about the company COPX. Is it an ETF?"
    gemini_cmd = "gemini" 
    # try standard paths if not in path (mimicking update_qual_features.py)
    paths = ["gemini", "/usr/local/bin/gemini", "C:\\Program Files\\Go\\bin\\gemini.exe"] 
    # Adjust paths for windows as user is on windows
    
    print(f"Running gemini with prompt: {prompt}")
    
    try:
        result = subprocess.run(
            ["gemini", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=120
        )
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
        print("--- RETURN CODE ---")
        print(result.returncode)
        
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    run_debug()
