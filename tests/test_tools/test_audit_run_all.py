import sys
from types import SimpleNamespace

import pytest

from scripts.audits import phase_1_automated, run_all


def test_run_all_quick_forwards_quick_to_quick_aware_phases(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, *, cwd, timeout):
        calls.append([str(part) for part in cmd])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_all.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--quick"])

    with pytest.raises(SystemExit) as excinfo:
        run_all.main()

    assert excinfo.value.code == 0
    phase_cmds = {cmd[1].split("\\")[-1]: cmd[2:] for cmd in calls}
    assert phase_cmds["phase_1_automated.py"] == ["--quick"]
    assert phase_cmds["phase_3_docs.py"] == ["--quick"]
    assert phase_cmds["phase_0_triage.py"] == []
    assert phase_cmds["phase_6_build_chain.py"] == []


def test_phase_1_quick_uses_bounded_pytest_and_drift_health_smoke(monkeypatch):
    calls: list[list[str]] = []

    def fake_run_tool(cmd, timeout=300):
        calls.append([str(part) for part in cmd])
        joined = " ".join(str(part) for part in cmd)
        if "-m pytest" in joined:
            return 0, "6 passed in 1.23s\n", ""
        if "py_compile pipeline/check_drift.py" in joined:
            return 0, "", ""
        if "pipeline.health_check" in joined:
            return 0, "[OK]\n[OK]\n", ""
        if "audit_integrity.py" in joined:
            return 0, "INTEGRITY AUDIT PASSED: all 10 checks clean\n", ""
        if "audit_behavioral.py" in joined:
            return 0, "BEHAVIORAL AUDIT PASSED: all 7 checks clean\n", ""
        return 0, "", ""

    monkeypatch.setattr(phase_1_automated, "_run_tool", fake_run_tool)
    monkeypatch.setattr(
        phase_1_automated, "_pytest_basetemp", lambda: phase_1_automated.PROJECT_ROOT / ".git" / "pytest-tmp" / "test"
    )
    monkeypatch.setattr(sys, "argv", ["phase_1_automated.py", "--quick"])

    with pytest.raises(SystemExit) as excinfo:
        phase_1_automated.main()

    assert excinfo.value.code == 0
    pytest_cmd = next(cmd for cmd in calls if "-m" in cmd and "pytest" in cmd)
    assert "tests/" not in pytest_cmd
    assert "-m" in pytest_cmd
    assert "not slow" in pytest_cmd
    assert "--basetemp" in pytest_cmd

    drift_cmd = next(cmd for cmd in calls if "pipeline/check_drift.py" in cmd)
    assert "-m" in drift_cmd
    assert "py_compile" in drift_cmd
    assert "--fast" not in drift_cmd
    assert "--quiet" not in drift_cmd

    health_cmd = next(cmd for cmd in calls if "pipeline.health_check" in cmd)
    assert "--quick" in health_cmd
