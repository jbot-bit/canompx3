"""
Register Windows Task Scheduler job to run daily Databento refresh at 7:00 AM Brisbane.
Brisbane = UTC+10, no DST. Run once: python scripts/setup_daily_backfill.py

7:00 AM Brisbane = 21:00 UTC previous day (markets close by 6:30 AM Brisbane at latest).
We schedule at 7:00 AM local (Brisbane machine time) so data is ready before the session.

Downloads: ohlcv-1m + ohlcv-1s + statistics + tbbo + trades + bbo-1s + mbp-1
(all covered by Standard plan subscription).
"""

import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
PYTHON = sys.executable
SCRIPT = PROJECT / "scripts" / "databento_daily.py"
TASK_NAME = "canompx3-daily-databento"
LOG_DIR = PROJECT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Build the command that the scheduler will run
# Use pythonw to avoid console window popup, with logging to file
run_cmd = f'"{PYTHON}" "{SCRIPT}"'

cmd = [
    "schtasks",
    "/create",
    "/f",  # Force overwrite if exists
    "/tn",
    TASK_NAME,
    "/tr",
    run_cmd,
    "/sc",
    "daily",
    "/st",
    "07:00",
    "/rl",
    "HIGHEST",
]

print(f"Registering scheduled task: {TASK_NAME}")
print("  Schedule: Daily at 07:00 Brisbane time")
print(f"  Script:   {SCRIPT}")
print(f"  Python:   {PYTHON}")
print(f"  Command:  {run_cmd}")
print()

r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode == 0:
    print(f"OK: Created task '{TASK_NAME}' -- runs daily at 07:00 Brisbane")
    print("  Covers: ohlcv-1m, ohlcv-1s, statistics, tbbo, trades, bbo-1s, mbp-1")
    print(f"  Log:    {LOG_DIR / 'databento_daily.log'}")
    print(f"\nVerify with: schtasks /query /tn {TASK_NAME}")
    print(f"Run now:     schtasks /run /tn {TASK_NAME}")
    print(f"Delete:      schtasks /delete /tn {TASK_NAME}")
else:
    print(f"FAIL: {r.stderr.strip()}")
    sys.exit(1)
