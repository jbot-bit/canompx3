"""Injection tests for Fast Lane Phase 5 capital-boundary drift check."""

from __future__ import annotations

from pathlib import Path

from pipeline.check_drift import check_fast_lane_phase5_capital_boundary


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_phase5_boundary_check_accepts_report_only_surface(tmp_path: Path) -> None:
    script = _write(
        tmp_path / "fast_lane_research_review.py",
        'CAPITAL_BOUNDARY = "REPORT_ONLY_NOT_DEPLOYMENT_AUTHORITY"\n'
        'RECOMMENDATIONS = ("KILL", "PARK", "BULLPEN", "RECOMMEND_RESEARCH_REVIEW", "ESCALATE_CAPITAL_REVIEW")\n',
    )

    assert check_fast_lane_phase5_capital_boundary(paths=(script,), report_script_path=script) == []


def test_phase5_boundary_check_rejects_deployment_candidate_wording(tmp_path: Path) -> None:
    script = _write(
        tmp_path / "fast_lane_research_review.py",
        'CAPITAL_BOUNDARY = "REPORT_ONLY_NOT_DEPLOYMENT_AUTHORITY"\nbad = "DEPLOYMENT_CANDIDATE"\n',
    )

    violations = check_fast_lane_phase5_capital_boundary(paths=(script,), report_script_path=script)

    assert violations
    assert "DEPLOYMENT_CANDIDATE" in violations[0]


def test_phase5_boundary_check_requires_boundary_token_in_reporter(tmp_path: Path) -> None:
    script = _write(tmp_path / "fast_lane_research_review.py", "RECOMMENDATIONS = ('PARK',)\n")

    violations = check_fast_lane_phase5_capital_boundary(paths=(script,), report_script_path=script)

    assert violations
    assert "REPORT_ONLY_NOT_DEPLOYMENT_AUTHORITY" in violations[0]


def test_phase5_boundary_check_rejects_capital_class_write_attempt(tmp_path: Path) -> None:
    script = _write(
        tmp_path / "fast_lane_research_review.py",
        'CAPITAL_BOUNDARY = "REPORT_ONLY_NOT_DEPLOYMENT_AUTHORITY"\n'
        'target = "docs/runtime/chordia_audit_log.yaml"\n'
        'Path(target).write_text("bad")\n',
    )

    violations = check_fast_lane_phase5_capital_boundary(paths=(script,), report_script_path=script)

    assert violations
    assert "CAPITAL-CLASS WRITE ATTEMPT" in "\n".join(violations)
