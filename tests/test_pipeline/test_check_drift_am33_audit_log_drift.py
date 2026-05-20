"""Injection tests for check_am33_audit_log_theory_grant_parity (Check #165).

Covers three violation directions:
  1. Prereg says theory_grant=True, audit log has no entry for that SID
     (implicit has_theory=False) → VIOLATION.
  2. Audit log says has_theory=True, prereg says theory_grant=False → VIOLATION.
  3. Audit log says has_theory=False (explicit), prereg says theory_grant=True → VIOLATION.

And the clean cases:
  4. Audit log has_theory=True, prereg theory_grant=True → no violation.
  5. Prereq has theory_grant=False, SID not in audit log → no violation.
  6. Drafts/ files are excluded from the scan.
"""

from __future__ import annotations

import textwrap

from pipeline.check_drift import check_am33_audit_log_theory_grant_parity


def _write_audit_log(tmp_path, theory_grants: list[dict] | None = None, default_has_theory: bool = False):
    payload = f"default_has_theory: {str(default_has_theory).lower()}\n"
    if theory_grants:
        payload += "theory_grants:\n"
        for entry in theory_grants:
            sid = entry["strategy_id"]
            ht = str(entry["has_theory"]).lower()
            ref = entry.get("theory_ref", "docs/institutional/literature/example.md")
            payload += f"  - strategy_id: '{sid}'\n    has_theory: {ht}\n    theory_ref: '{ref}'\n"
    else:
        payload += "theory_grants: []\n"
    p = tmp_path / "chordia_audit_log.yaml"
    p.write_text(payload, encoding="utf-8")
    return p


def _write_prereg(hyp_dir, slug: str, strategy_id: str, theory_grant: bool):
    content = textwrap.dedent(f"""\
        metadata:
          theory_grant: {str(theory_grant).lower()}
        scope:
          strategy_id: '{strategy_id}'
    """)
    p = hyp_dir / f"{slug}.yaml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Violation: prereg=true, audit log has NO entry for SID (implicit false)
# ---------------------------------------------------------------------------


def test_violation_prereg_true_audit_log_absent(tmp_path):
    """Prereg declares theory_grant=true but SID is absent from audit log theory_grants."""
    audit_path = _write_audit_log(tmp_path, theory_grants=[])
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    _write_prereg(hyp_dir, "2026-01-01-test-prereg", "TEST_STRATEGY_ID_ABC", theory_grant=True)

    violations = check_am33_audit_log_theory_grant_parity(audit_log_path=audit_path, hypotheses_dir=hyp_dir)
    assert violations, "Expected a violation but got none"
    combined = "\n".join(violations)
    assert "PARITY MISMATCH" in combined
    assert "TEST_STRATEGY_ID_ABC" in combined
    assert "theory_grant=true" in combined
    assert "chordia_audit_log.yaml has NO" in combined


# ---------------------------------------------------------------------------
# Violation: audit log has_theory=true, prereg theory_grant=false
# ---------------------------------------------------------------------------


def test_violation_audit_log_true_prereg_false(tmp_path):
    """Audit log says has_theory=true but prereg declares theory_grant=false."""
    audit_path = _write_audit_log(
        tmp_path, theory_grants=[{"strategy_id": "TEST_SID_XYZ", "has_theory": True, "theory_ref": "docs/example.md"}]
    )
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    _write_prereg(hyp_dir, "2026-01-02-test-prereg", "TEST_SID_XYZ", theory_grant=False)

    violations = check_am33_audit_log_theory_grant_parity(audit_log_path=audit_path, hypotheses_dir=hyp_dir)
    assert violations, "Expected a violation but got none"
    combined = "\n".join(violations)
    assert "PARITY MISMATCH" in combined
    assert "TEST_SID_XYZ" in combined


# ---------------------------------------------------------------------------
# Violation: audit log explicit has_theory=false, prereg theory_grant=true
# ---------------------------------------------------------------------------


def test_violation_audit_log_explicit_false_prereg_true(tmp_path):
    """Audit log has an explicit has_theory=false entry; prereg says theory_grant=true."""
    audit_path = _write_audit_log(
        tmp_path, theory_grants=[{"strategy_id": "TEST_SID_EXP", "has_theory": False, "theory_ref": ""}]
    )
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    _write_prereg(hyp_dir, "2026-01-03-test-prereg", "TEST_SID_EXP", theory_grant=True)

    violations = check_am33_audit_log_theory_grant_parity(audit_log_path=audit_path, hypotheses_dir=hyp_dir)
    assert violations, "Expected a violation but got none"
    combined = "\n".join(violations)
    assert "PARITY MISMATCH" in combined
    assert "TEST_SID_EXP" in combined


# ---------------------------------------------------------------------------
# Clean: audit log has_theory=true, prereg theory_grant=true
# ---------------------------------------------------------------------------


def test_clean_both_true(tmp_path):
    """Both surfaces agree has_theory=true → no violation."""
    audit_path = _write_audit_log(
        tmp_path,
        theory_grants=[
            {"strategy_id": "CLEAN_SID_TRUE", "has_theory": True, "theory_ref": "docs/institutional/literature/chan.md"}
        ],
    )
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    _write_prereg(hyp_dir, "2026-01-04-clean-prereg", "CLEAN_SID_TRUE", theory_grant=True)

    violations = check_am33_audit_log_theory_grant_parity(audit_log_path=audit_path, hypotheses_dir=hyp_dir)
    assert violations == [], f"Unexpected violations: {violations}"


# ---------------------------------------------------------------------------
# Clean: prereg theory_grant=false, SID not in audit log
# ---------------------------------------------------------------------------


def test_clean_prereg_false_not_in_audit_log(tmp_path):
    """Prereg theory_grant=false, SID absent from audit log → no violation (correct default)."""
    audit_path = _write_audit_log(tmp_path, theory_grants=[])
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    _write_prereg(hyp_dir, "2026-01-05-clean-prereg", "CLEAN_SID_FALSE", theory_grant=False)

    violations = check_am33_audit_log_theory_grant_parity(audit_log_path=audit_path, hypotheses_dir=hyp_dir)
    assert violations == [], f"Unexpected violations: {violations}"


# ---------------------------------------------------------------------------
# Drafts are excluded from the scan
# ---------------------------------------------------------------------------


def test_drafts_excluded(tmp_path):
    """Files in a 'drafts' subdirectory are not scanned."""
    audit_path = _write_audit_log(tmp_path, theory_grants=[])
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    # Write the "violation" file into drafts/ — should be invisible to the check.
    drafts_dir = hyp_dir / "drafts"
    drafts_dir.mkdir()
    _write_prereg(drafts_dir, "2026-01-06-draft-prereg", "DRAFT_SID_ONLY", theory_grant=True)

    violations = check_am33_audit_log_theory_grant_parity(audit_log_path=audit_path, hypotheses_dir=hyp_dir)
    assert violations == [], f"Unexpected violations from drafts/: {violations}"


# ---------------------------------------------------------------------------
# Fail-closed: missing audit log returns violation
# ---------------------------------------------------------------------------


def test_fail_closed_missing_audit_log(tmp_path):
    """Missing chordia_audit_log.yaml returns a violation rather than silently passing."""
    audit_path = tmp_path / "nonexistent_audit_log.yaml"
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()

    violations = check_am33_audit_log_theory_grant_parity(audit_log_path=audit_path, hypotheses_dir=hyp_dir)
    assert violations, "Missing audit log should produce a violation"
    combined = "\n".join(violations)
    assert "chordia_audit_log.yaml not found" in combined
