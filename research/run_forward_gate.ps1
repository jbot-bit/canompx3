$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

python research/update_forward_gate_tracker.py

Write-Host "`n--- forward_gate_status_latest.md ---`n"
Get-Content -Raw research/output/forward_gate_status_latest.md
