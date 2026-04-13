from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.tools import context_views


def test_build_research_context_uses_narrow_research_surfaces(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(context_views, "_canonical_repo_root", lambda root: root)
    monkeypatch.setattr(context_views, "_git_branch", lambda root: "feature/context")
    monkeypatch.setattr(context_views, "_git_head", lambda root: "abc123")
    monkeypatch.setattr(context_views, "collect_system_identity", lambda root, canonical, db_path: ({"ok": True}, []))
    monkeypatch.setattr(
        context_views,
        "collect_fitness_fast",
        lambda db_path: ({"MES": {"active_strategies": 3}}, []),
    )
    monkeypatch.setattr(
        context_views,
        "collect_handoff",
        lambda root: ({"date": "2026-04-13", "summary": "Investigate MES", "next_steps": ["Query DB"]}, []),
    )

    payload = context_views.build_research_context(tmp_path, tmp_path / "gold.db")

    assert payload["view"] == "research"
    assert payload["sections"]["live_operational_state"]["fitness_summary"]["MES"]["active_strategies"] == 3
    assert payload["sections"]["live_operational_state"]["repo_runtime_context"]["available"] is True
    assert (
        payload["sections"]["canonical_state"]["holdout_policy"]["canonical_source"] == "trading_app/holdout_policy.py"
    )
    assert payload["sections"]["non_authoritative_context"]["handoff"]["summary"] == "Investigate MES"


def test_build_trading_context_uses_runtime_collectors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(context_views, "_canonical_repo_root", lambda root: root)
    monkeypatch.setattr(context_views, "_git_branch", lambda root: "feature/context")
    monkeypatch.setattr(context_views, "_git_head", lambda root: "abc123")
    monkeypatch.setattr(context_views, "collect_system_identity", lambda root, canonical, db_path: ({"ok": True}, []))
    monkeypatch.setattr(
        context_views,
        "collect_deployment_state",
        lambda db_path: ({"profile_id": "topstep", "deployed_count": 2, "validated_active_count": 4}, []),
    )
    monkeypatch.setattr(
        context_views,
        "collect_lifecycle_control",
        lambda db_path: (
            {"gate_ok": True},
            {"available": True},
            {"paused_count": 1},
            [],
        ),
    )
    monkeypatch.setattr(
        context_views,
        "collect_upcoming_sessions",
        lambda db_path: [{"label": "NY", "brisbane_time": "22:30", "hours_away": 2.0}],
    )

    payload = context_views.build_trading_context(tmp_path, tmp_path / "gold.db")

    assert payload["view"] == "trading"
    live = payload["sections"]["live_operational_state"]
    assert live["deployment_summary"]["deployed_count"] == 2
    assert live["pause_summary"]["paused_count"] == 1
    assert live["upcoming_sessions"][0]["label"] == "NY"
    assert live["repo_runtime_context"]["available"] is True
    assert payload["sections"]["non_authoritative_context"] == {}


def test_build_recent_performance_context_projects_strategy_fitness(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(context_views, "_canonical_repo_root", lambda root: root)
    monkeypatch.setattr(context_views, "_git_branch", lambda root: "feature/context")
    monkeypatch.setattr(context_views, "_git_head", lambda root: "abc123")
    monkeypatch.setattr(context_views, "collect_system_identity", lambda root, canonical, db_path: ({"ok": True}, []))
    monkeypatch.setattr(context_views, "ACTIVE_ORB_INSTRUMENTS", {"MES"})

    fake_report = SimpleNamespace(
        as_of_date=SimpleNamespace(isoformat=lambda: "2026-04-13"),
        summary={"fit": 1, "watch": 1, "decay": 0, "stale": 0},
        scores=[
            SimpleNamespace(
                strategy_id="MES_A",
                fitness_status="WATCH",
                fitness_notes="Declining Sharpe",
                rolling_window_months=3,
                rolling_exp_r=0.04,
                rolling_sharpe=0.3,
                rolling_win_rate=0.42,
                rolling_sample=18,
                recent_sharpe_30=-0.2,
                recent_sharpe_60=0.1,
                sharpe_delta_30=-0.4,
                sharpe_delta_60=-0.1,
            ),
            SimpleNamespace(
                strategy_id="MES_B",
                fitness_status="FIT",
                fitness_notes="Stable",
                rolling_window_months=3,
                rolling_exp_r=0.12,
                rolling_sharpe=0.8,
                rolling_win_rate=0.55,
                rolling_sample=30,
                recent_sharpe_30=0.2,
                recent_sharpe_60=0.3,
                sharpe_delta_30=0.05,
                sharpe_delta_60=0.02,
            ),
        ],
    )
    monkeypatch.setattr(
        context_views,
        "_collect_recent_performance",
        lambda instruments, db_path: {
            "available": True,
            "rolling_window_months": context_views.RECENT_PERFORMANCE_ROLLING_MONTHS,
            "instrument_reports": {
                "MES": {
                    "as_of_date": fake_report.as_of_date.isoformat(),
                    "strategy_count": len(fake_report.scores),
                    "summary": fake_report.summary,
                    "non_fit_count": 1,
                    "non_fit_strategies": [context_views._project_fitness_score(fake_report.scores[0])],
                    "lowest_recent_sharpe_30": [context_views._project_fitness_score(fake_report.scores[0])],
                    "lowest_rolling_exp_r": [context_views._project_fitness_score(fake_report.scores[0])],
                }
            },
            "instrument_errors": {},
        },
    )

    payload = context_views.build_recent_performance_context(tmp_path, tmp_path / "gold.db")

    assert payload["view"] == "recent_performance"
    assert payload["sections"]["canonical_state"]["recent_performance_contract"]["canonical_source"] == (
        "trading_app/strategy_fitness.py"
    )
    live = payload["sections"]["live_operational_state"]
    assert live["recent_performance"]["available"] is True
    assert live["recent_performance"]["instrument_reports"]["MES"]["non_fit_count"] == 1
    assert (
        live["recent_performance"]["instrument_reports"]["MES"]["lowest_recent_sharpe_30"][0]["strategy_id"] == "MES_A"
    )
    assert payload["sections"]["non_authoritative_context"] == {}


def test_build_verification_context_uses_repo_state_collectors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(context_views, "_canonical_repo_root", lambda root: root)
    monkeypatch.setattr(context_views, "_git_branch", lambda root: "feature/context")
    monkeypatch.setattr(context_views, "_git_head", lambda root: "abc123")
    monkeypatch.setattr(context_views, "collect_system_identity", lambda root, canonical, db_path: ({"ok": True}, []))
    monkeypatch.setattr(
        context_views,
        "collect_handoff",
        lambda root: ({"date": "2026-04-13", "summary": "Verify slice", "next_steps": ["Run drift"]}, []),
    )
    monkeypatch.setattr(
        context_views,
        "collect_session_claims",
        lambda root: [],
    )

    payload = context_views.build_verification_context(tmp_path, tmp_path / "gold.db")

    assert payload["view"] == "verification"
    assert payload["sections"]["non_authoritative_context"]["handoff"]["summary"] == "Verify slice"
    commands = payload["sections"]["canonical_state"]["verification_profile"]["commands"]
    assert payload["sections"]["live_operational_state"]["session_claim_item_count"] == 0
    assert any("pipeline/check_drift.py" in command for command in commands)


def test_validate_view_payload_rejects_mixed_truth() -> None:
    violations = context_views.validate_view_payload(
        {
            "sections": {
                "canonical_state": {"handoff": {"summary": "wrong"}, "recommendation": "bad"},
                "live_operational_state": {"signals": {"bad": 1}},
                "non_authoritative_context": {},
            },
            "section_sources": {
                "canonical_state": ["x"],
                "live_operational_state": ["y"],
                "non_authoritative_context": ["z"],
            },
        }
    )

    assert "handoff context leaked into canonical_state" in violations
    assert "recommendation is not allowed in canonical_state" in violations
    assert "signals is not allowed in live_operational_state" in violations


def test_build_view_uses_registered_view_builders(monkeypatch, tmp_path: Path) -> None:
    sentinel = {
        "sections": {"canonical_state": {}, "live_operational_state": {}, "non_authoritative_context": {}},
        "section_sources": {"canonical_state": [], "live_operational_state": [], "non_authoritative_context": []},
    }
    monkeypatch.setitem(context_views.VIEW_BUILDERS, "research", lambda root, db_path: sentinel)

    payload = context_views.build_view("research", tmp_path, tmp_path / "gold.db")

    assert payload is sentinel
