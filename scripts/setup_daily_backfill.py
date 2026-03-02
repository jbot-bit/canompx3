"""
Register Windows Task Scheduler job to run daily backfill at 7:00 AM Brisbane.
Brisbane = UTC+10, no DST. Run once: python scripts/setup_daily_backfill.py

7:00 AM Brisbane = 21:00 UTC previous day (markets close by 6:30 AM Brisbane at latest).
We schedule at 7:00 AM local (Brisbane machine time) so data is ready before the session.
"""
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
PYTHON = sys.executable
SCRIPT = PROJECT / "pipeline" / "daily_backfill.py"
TASK_NAME = "canompx3-daily-backfill"

cmd = [
    "schtasks", "/create", "/f",
    "/tn", TASK_NAME,
    "/tr", f'"{PYTHON}" "{SCRIPT}"',
    "/sc", "daily",
    "/st", "07:00",
    "/rl", "HIGHEST",
]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode == 0:
    print(f"✓ Created task '{TASK_NAME}' — runs daily at 07:00 Brisbane")
    print(f"  Script: {SCRIPT}")
    print(f"  Python: {PYTHON}")
    print(f"\nVerify with: schtasks /query /tn {TASK_NAME}")
else:
    print(f"✗ Failed to create task: {r.stderr.strip()}")
    sys.exit(1)
