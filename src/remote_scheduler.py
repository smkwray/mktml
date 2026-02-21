import time
import datetime
import subprocess
import sys
import os

# Prevent __pycache__ generation (helps cloud sync churn).
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True

def schedule_scan(target_hour=9, target_minute=30):
    """
    Sleeps until the next occurrence of target time, then runs the scanner.
    """
    now = datetime.datetime.now()
    target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    
    if target <= now:
        # If target has passed today, schedule for tomorrow
        target += datetime.timedelta(days=1)
        
    delay = (target - now).total_seconds()
    
    print(f"[{now}] Scheduler started.")
    print(f"Target time: {target}")
    print(f"Time until scan: {datetime.timedelta(seconds=delay)}")
    print("Going to sleep...", flush=True)
    
    time.sleep(delay)
    
    print(f"\n[{datetime.datetime.now()}] Waking up! Starting scan...", flush=True)
    
    # Run the scanner
    # Assumes running from project root
    cmd = [sys.executable, "main.py", "--scan"]
    
    with open("scan_scheduled.log", "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        
    print(f"[{datetime.datetime.now()}] Scan complete. Output logged to scan_scheduled.log")

if __name__ == "__main__":
    schedule_scan(9, 30)
