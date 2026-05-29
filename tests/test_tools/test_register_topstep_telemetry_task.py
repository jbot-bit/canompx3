from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "tools" / "register_topstep_telemetry_task.ps1"


def test_topstep_telemetry_task_installer_contract() -> None:
    """Scheduler installer must recreate the signal-only telemetry task safely."""
    text = SCRIPT.read_text(encoding="utf-8")

    assert "CanonMPX_TopstepTelemetry_SignalOnly" in text
    assert "topstep_50k_mnq_auto" in text
    assert "--signal-only" in text
    assert "--demo" not in text
    assert "--live" not in text
    assert "CANOMPX3_DASHBOARD_ORIGIN=1" in text
    assert '[string]$DuckDbPath = "C:\\Users\\joshd\\canompx3\\gold.db"' in text
    assert "set DUCKDB_PATH=$DuckDbPath" in text
    assert "MultipleInstances IgnoreNew" in text
    assert "DisallowStartIfOnBatteries" not in text
    assert "StopIfGoingOnBatteries" not in text
    assert "Register-ScheduledTask" in text
    assert "logs\\telemetry_accrual_signal_only.log" in text
