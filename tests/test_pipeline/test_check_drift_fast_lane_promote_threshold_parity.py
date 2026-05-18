"""Injection tests for check_fast_lane_promote_threshold_parity (Check #158).

Mutation-probe per constant — sibling coverage per
[[regex-alternation-sibling-coverage]] — flipping any of the six gated
constants on the imported scanner module must produce exactly one violation
that names that constant, the observed value, and the canonical value.

Constants covered by dedicated injection (one test per):
  - T_KILL_FLOOR
  - T_PROMOTE_FLOOR
  - EXPR_FLOOR
  - N_FLOOR
  - FIRE_MIN
  - FIRE_MAX

Plus fail-closed coverage for malformed-template paths (missing file,
``key: null``, non-numeric value) — auditor-driven additions per
``adversarial-audit-gate.md`` 2026-05-19 round.

Class anchor: [[canonical-inline-copy-parity-bug-class]] (4th confirmed
instance, 2026-05-19).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.check_drift import check_fast_lane_promote_threshold_parity
from scripts.research import fast_lane_promote_queue as flpq


# ----------------------------- clean-state ------------------------------


def test_clean_state_passes():
    """Real scanner constants vs real canonical template must match."""
    violations = check_fast_lane_promote_threshold_parity()
    assert violations == [], f"unexpected parity violations: {violations}"


# --------------------------- fail-closed --------------------------------


def test_missing_template_fails_closed(tmp_path: Path):
    """If the canonical template is unreachable the check returns a single
    violation rather than silently passing."""
    forged = tmp_path / "does-not-exist.yaml"
    violations = check_fast_lane_promote_threshold_parity(template_path=forged)
    assert violations, "missing canonical template must NOT pass silently"
    assert len(violations) == 1
    assert "canonical template missing" in violations[0]
    assert str(forged) in violations[0]


# ---------------------- per-constant injection --------------------------


def test_drift_in_T_KILL_FLOOR_is_caught(monkeypatch: pytest.MonkeyPatch):
    """Flip scanner T_KILL_FLOOR away from canonical 2.5 — must violate."""
    monkeypatch.setattr(flpq, "T_KILL_FLOOR", 2.0)
    violations = check_fast_lane_promote_threshold_parity()
    assert violations, "T_KILL_FLOOR drift went undetected"
    relevant = [v for v in violations if "T_KILL_FLOOR" in v]
    assert relevant, f"no violation mentioned T_KILL_FLOOR: {violations}"
    assert "2.0" in relevant[0]
    assert "2.5" in relevant[0]
    assert "screen.promote_threshold" in relevant[0]


def test_drift_in_T_PROMOTE_FLOOR_is_caught(monkeypatch: pytest.MonkeyPatch):
    """Flip scanner T_PROMOTE_FLOOR away from canonical 3.0 — must violate."""
    monkeypatch.setattr(flpq, "T_PROMOTE_FLOOR", 2.75)
    violations = check_fast_lane_promote_threshold_parity()
    assert violations, "T_PROMOTE_FLOOR drift went undetected"
    relevant = [v for v in violations if "T_PROMOTE_FLOOR" in v]
    assert relevant, f"no violation mentioned T_PROMOTE_FLOOR: {violations}"
    assert "2.75" in relevant[0]
    assert "3.0" in relevant[0]
    assert "promote_threshold + screen.needs_more_band" in relevant[0]


def test_drift_in_EXPR_FLOOR_is_caught(monkeypatch: pytest.MonkeyPatch):
    """Flip scanner EXPR_FLOOR away from canonical 0.0 — must violate.

    Closes sibling-coverage gap surfaced by 2026-05-19 evidence-auditor
    pass: EXPR_FLOOR was in the parity check but had no dedicated
    mutation-probe per [[regex-alternation-sibling-coverage]].
    """
    monkeypatch.setattr(flpq, "EXPR_FLOOR", 0.1)
    violations = check_fast_lane_promote_threshold_parity()
    assert violations, "EXPR_FLOOR drift went undetected"
    relevant = [v for v in violations if "EXPR_FLOOR" in v]
    assert relevant, f"no violation mentioned EXPR_FLOOR: {violations}"
    assert "0.1" in relevant[0]
    assert "0.0" in relevant[0]
    assert "screen.expr_min" in relevant[0]


def test_drift_in_N_FLOOR_is_caught(monkeypatch: pytest.MonkeyPatch):
    """Flip scanner N_FLOOR away from canonical 50 — must violate."""
    monkeypatch.setattr(flpq, "N_FLOOR", 30)
    violations = check_fast_lane_promote_threshold_parity()
    assert violations, "N_FLOOR drift went undetected"
    relevant = [v for v in violations if "N_FLOOR" in v]
    assert relevant, f"no violation mentioned N_FLOOR: {violations}"
    assert "30" in relevant[0]
    assert "50" in relevant[0]
    assert "screen.n_IS_on_min" in relevant[0]


def test_drift_in_FIRE_MIN_is_caught(monkeypatch: pytest.MonkeyPatch):
    """Flip scanner FIRE_MIN away from canonical 0.05 — must violate."""
    monkeypatch.setattr(flpq, "FIRE_MIN", 0.10)
    violations = check_fast_lane_promote_threshold_parity()
    assert violations, "FIRE_MIN drift went undetected"
    relevant = [v for v in violations if "FIRE_MIN" in v]
    assert relevant, f"no violation mentioned FIRE_MIN: {violations}"
    assert "0.1" in relevant[0]
    assert "0.05" in relevant[0]
    assert "kill_if" in relevant[0]


def test_drift_in_FIRE_MAX_is_caught(monkeypatch: pytest.MonkeyPatch):
    """Flip scanner FIRE_MAX away from canonical 0.95 — must violate."""
    monkeypatch.setattr(flpq, "FIRE_MAX", 0.99)
    violations = check_fast_lane_promote_threshold_parity()
    assert violations, "FIRE_MAX drift went undetected"
    relevant = [v for v in violations if "FIRE_MAX" in v]
    assert relevant, f"no violation mentioned FIRE_MAX: {violations}"
    assert "0.99" in relevant[0]
    assert "0.95" in relevant[0]
    assert "kill_if" in relevant[0]


# ---------------------- template-side drift ------------------------------


def test_null_canonical_value_fails_closed(tmp_path: Path):
    """``promote_threshold:`` with no value parses as None in YAML — must
    return a structural violation, NOT crash on ``float(None)``.

    Regression test for 2026-05-19 evidence-auditor finding (CONDITIONAL
    verdict): _require() previously distinguished only "key absent" and
    silently returned None for "key present, value null", which then
    crashed downstream casts.
    """
    forged = tmp_path / "TEMPLATE-fast-lane-v5.1.yaml"
    forged.write_text(
        """\
screen:
  metric: t_IS
  promote_threshold:
  expr_min: 0.0
  n_IS_on_min: 50
  needs_more_band: 0.5
  fire_rate_gate:
    kill_if: "fire_rate < 0.05 OR fire_rate > 0.95"
""",
        encoding="utf-8",
    )
    violations = check_fast_lane_promote_threshold_parity(template_path=forged)
    assert violations, "null canonical value must return a structural violation"
    assert any("promote_threshold" in v for v in violations)
    assert any("NoneType" in v or "expected numeric" in v for v in violations)


def test_non_numeric_canonical_value_fails_closed(tmp_path: Path):
    """``promote_threshold: [2.5]`` (accidentally a list) — must fail closed,
    NOT crash on ``float([2.5])``.

    Same class as the null case; auditor flagged the unguarded cast at
    pipeline/check_drift.py float()/int() invocations.
    """
    forged = tmp_path / "TEMPLATE-fast-lane-v5.1.yaml"
    forged.write_text(
        """\
screen:
  metric: t_IS
  promote_threshold: [2.5]
  expr_min: 0.0
  n_IS_on_min: 50
  needs_more_band: 0.5
  fire_rate_gate:
    kill_if: "fire_rate < 0.05 OR fire_rate > 0.95"
""",
        encoding="utf-8",
    )
    violations = check_fast_lane_promote_threshold_parity(template_path=forged)
    assert violations, "list-typed canonical value must return a structural violation"
    assert any("promote_threshold" in v and "list" in v for v in violations)


def test_drift_template_amend_caught(tmp_path: Path):
    """If the canonical template's promote_threshold moves but the scanner
    is not updated, the check must catch the mismatch from the template
    side. Confirms the canonical-source reload path is live, not cached."""
    forged = tmp_path / "TEMPLATE-fast-lane-v5.1.yaml"
    forged.write_text(
        """\
screen:
  metric: t_IS
  promote_threshold: 2.0
  expr_min: 0.0
  n_IS_on_min: 50
  needs_more_band: 0.5
  fire_rate_gate:
    kill_if: "fire_rate < 0.05 OR fire_rate > 0.95"
""",
        encoding="utf-8",
    )
    violations = check_fast_lane_promote_threshold_parity(template_path=forged)
    assert violations, "template-side promote_threshold drift went undetected"
    # Scanner T_KILL_FLOOR=2.5 vs forged template 2.0 -> T_KILL_FLOOR violation;
    # T_PROMOTE_FLOOR=3.0 vs forged 2.0+0.5=2.5 -> T_PROMOTE_FLOOR violation.
    assert any("T_KILL_FLOOR" in v for v in violations)
    assert any("T_PROMOTE_FLOOR" in v for v in violations)
