@echo off
cd /d %~dp0\..
python research\update_forward_gate_tracker.py
if exist research\output\forward_gate_status_latest.md (
  echo.
  echo --- forward_gate_status_latest.md ---
  type research\output\forward_gate_status_latest.md
)
