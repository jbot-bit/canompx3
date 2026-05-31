from __future__ import annotations

import json
from pathlib import Path

from scripts.tools import capital_hard_audit
from scripts.tools.project_pulse import PulseItem, PulseReport


def _mkfile(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _pulse(*items: PulseItem) -> PulseReport:
    return PulseReport(
        generated_at="2026-05-31T00:00:00+00:00",
        cache_hit=False,
        git_head="abc1234",
        git_branch="main",
        items=list(items),
    )


def test_blocks_on_live_readiness_blockers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        capital_hard_audit,
        "build_live_readiness_report",
        lambda **_kwargs: {
            "profile_id": "topstep_50k_mnq_auto",
            "strict_zero_warn": {"green": False, "blockers": ["telemetry maturity missing"], "warnings": []},
        },
    )
    monkeypatch.setattr(capital_hard_audit, "build_pulse", lambda *args, **kwargs: _pulse())
    monkeypatch.setattr(capital_hard_audit, "_git_context", lambda _root: {"branch": "main", "head": "abc1234"})

    report = capital_hard_audit.build_capital_hard_audit(
        decision_target="promote topstep_50k_mnq_auto",
        role="filter",
        object_unit="strategy",
        horizon="pre-trade",
        root=tmp_path,
    )

    assert report["verdict"] == "BLOCK"
    assert "telemetry maturity missing" in "\n".join(report["blockers"])


def test_same_evidence_downgrades_standalone_without_alternative_framing(tmp_path: Path, monkeypatch) -> None:
    claim = tmp_path / "docs" / "audit" / "results" / "claim.md"
    _mkfile(
        claim,
        "\n".join(
            [
                "# Claim",
                "",
                "Disposition: DEAD_FOR_ORB",
                "",
                "Verified with pytest and check_drift.",
            ]
        ),
    )
    monkeypatch.setattr(
        capital_hard_audit,
        "build_live_readiness_report",
        lambda **_kwargs: {
            "profile_id": "topstep_50k_mnq_auto",
            "strict_zero_warn": {"green": True, "blockers": [], "warnings": []},
        },
    )
    monkeypatch.setattr(capital_hard_audit, "build_pulse", lambda *args, **kwargs: _pulse())
    monkeypatch.setattr(capital_hard_audit, "_git_context", lambda _root: {"branch": "main", "head": "abc1234"})

    report = capital_hard_audit.build_capital_hard_audit(
        decision_target="decide if mechanism is dead",
        role="standalone",
        object_unit="mechanism",
        horizon="retrospective",
        claim_paths=[claim],
        root=tmp_path,
    )

    assert report["verdict"] == "VERIFY_MORE"
    assert any("alternative framing" in defect.lower() for defect in report["framing_defects"])


def test_shadow_only_path_can_accept_risk_when_framing_is_explicit(tmp_path: Path, monkeypatch) -> None:
    claim = tmp_path / "docs" / "audit" / "results" / "shadow.md"
    _mkfile(
        claim,
        "\n".join(
            [
                "# Shadow-only path",
                "",
                "## Alternative Framing Check",
                "- standalone",
                "- shadow-only",
                "- allocator",
                "",
                "## What would falsify this verdict",
                "- live shadow divergence",
                "",
                "## Unchecked scope",
                "- no deployability claim",
                "",
                "## Limitations",
                "- signal-only path",
            ]
        ),
    )
    monkeypatch.setattr(
        capital_hard_audit,
        "build_live_readiness_report",
        lambda **_kwargs: {
            "profile_id": "topstep_50k_mnq_auto",
            "strict_zero_warn": {"green": True, "blockers": [], "warnings": ["automation health not green"]},
        },
    )
    monkeypatch.setattr(capital_hard_audit, "build_pulse", lambda *args, **kwargs: _pulse())
    monkeypatch.setattr(capital_hard_audit, "_git_context", lambda _root: {"branch": "main", "head": "abc1234"})

    report = capital_hard_audit.build_capital_hard_audit(
        decision_target="shadow-only SR-monitor path",
        role="shadow-only",
        object_unit="strategy",
        horizon="post-trigger",
        claim_paths=[claim],
        root=tmp_path,
    )

    assert report["verdict"] == "ACCEPT_WITH_RISK"
    assert report["accepted_risks"]


def test_clear_requires_framing_and_falsification_sections(tmp_path: Path, monkeypatch) -> None:
    claim = tmp_path / "docs" / "audit" / "results" / "clear.md"
    _mkfile(
        claim,
        "\n".join(
            [
                "# Capital claim",
                "",
                "## Alternative Framings Checked",
                "- filter",
                "- allocator",
                "- diagnostic",
                "",
                "## What would falsify this verdict",
                "- failing strict-zero-warn",
                "- broken pulse follow-up coverage",
                "",
                "## Unchecked scope",
                "- no replacement-role decision in this pass",
                "",
                "Verification: pytest, check_drift, canonical evidence.",
            ]
        ),
    )
    monkeypatch.setattr(
        capital_hard_audit,
        "build_live_readiness_report",
        lambda **_kwargs: {
            "profile_id": "topstep_50k_mnq_auto",
            "strict_zero_warn": {"green": True, "blockers": [], "warnings": []},
        },
    )
    monkeypatch.setattr(capital_hard_audit, "build_pulse", lambda *args, **kwargs: _pulse())
    monkeypatch.setattr(capital_hard_audit, "_git_context", lambda _root: {"branch": "main", "head": "abc1234"})

    report = capital_hard_audit.build_capital_hard_audit(
        decision_target="deployability decision",
        role="filter",
        object_unit="strategy",
        horizon="pre-trade",
        claim_paths=[claim],
        root=tmp_path,
    )

    assert report["verdict"] == "CLEAR"
    assert report["alternative_framings"]
    assert report["unchecked_scope"]


def test_json_output_contains_object_and_framing_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        capital_hard_audit,
        "build_live_readiness_report",
        lambda **_kwargs: {
            "profile_id": "topstep_50k_mnq_auto",
            "strict_zero_warn": {"green": False, "blockers": ["x"], "warnings": []},
        },
    )
    monkeypatch.setattr(capital_hard_audit, "build_pulse", lambda *args, **kwargs: _pulse())
    monkeypatch.setattr(capital_hard_audit, "_git_context", lambda _root: {"branch": "main", "head": "abc1234"})

    report = capital_hard_audit.build_capital_hard_audit(
        decision_target="promote lane",
        role="allocator",
        object_unit="profile",
        horizon="portfolio",
        root=tmp_path,
    )
    payload = json.loads(capital_hard_audit.render_json(report))

    assert payload["decision_target"] == "promote lane"
    assert payload["object"]["role"] == "allocator"
    assert "alternative_framings" in payload
    assert "unchecked_scope" in payload
