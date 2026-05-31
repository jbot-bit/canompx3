from __future__ import annotations

from types import SimpleNamespace

from scripts.tools import start_topstep_live_pilot as pilot


def test_gate_plan_pins_single_copy_live_pilot():
    config = pilot.LivePilotConfig()

    steps = pilot.build_gate_steps(config)
    launch = pilot.build_launch_step(config)

    assert [step.name for step in steps] == [
        "Refresh C11/C12 control state",
        "Strict live readiness",
        "Live-session preflight",
    ]
    readiness = steps[1].argv
    preflight = steps[2].argv
    launch_argv = launch.argv

    assert "--profile" in readiness
    assert "topstep_50k_mnq_auto" in readiness
    assert "--copies" in readiness
    assert "1" in readiness
    assert "--strict-zero-warn" in readiness

    assert "--instrument" in preflight
    assert "MNQ" in preflight
    assert "--live" in preflight
    assert "--copies" in preflight
    assert "1" in preflight
    assert "--preflight" in preflight

    assert "--live" in launch_argv
    assert "--copies" in launch_argv
    assert "1" in launch_argv
    assert "--auto-confirm" not in launch_argv
    assert "--preflight" not in launch_argv


def test_skip_control_refresh_removes_refresh_step():
    steps = pilot.build_gate_steps(pilot.LivePilotConfig(), refresh_control_state=False)

    assert [step.name for step in steps] == ["Strict live readiness", "Live-session preflight"]


def test_run_live_pilot_short_circuits_on_failed_gate(monkeypatch):
    calls: list[str] = []

    def fake_run(argv, cwd):  # noqa: ANN001
        calls.append(tuple(argv)[1])
        if calls[-1].endswith("refresh_control_state.py"):
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pilot.subprocess, "run", fake_run)

    rc = pilot.run_live_pilot()

    assert rc == 1
    assert calls == ["scripts/tools/refresh_control_state.py"]


def test_preflight_only_does_not_launch(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fake_run(argv, cwd):  # noqa: ANN001
        calls.append(tuple(argv))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pilot.subprocess, "run", fake_run)

    rc = pilot.run_live_pilot(preflight_only=True)

    assert rc == 0
    assert len(calls) == 3
    assert all("--preflight" in call or call[1].endswith(("refresh_control_state.py", "live_readiness_report.py")) for call in calls)


def test_dry_run_never_invokes_subprocess(monkeypatch, capsys):
    def fake_run(*_args, **_kwargs):
        raise AssertionError("dry-run must not invoke subprocess")

    monkeypatch.setattr(pilot.subprocess, "run", fake_run)

    rc = pilot.main(["--dry-run"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Topstep MNQ live pilot command plan" in out
    assert "--auto-confirm" not in out
