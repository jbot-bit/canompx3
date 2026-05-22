"""Injection tests for check_amendment_3_4_provisional_gate.

Amendment 3.4 (PROVISIONAL, 2026-05-23) at
``docs/institutional/pre_registered_criteria.md`` blocks new
mechanism-class-transfer ``theory_grant: true`` preregs until the re-audit
closes. The drift check enforces it at commit time.

Cases covered:
  1. Post-sentinel prereg, theory_grant=true, no escape field -> BLOCK.
  2. Post-sentinel prereg, theory_grant=true, re-audit-closed escape -> PASS.
  3. Post-sentinel prereg, theory_grant=false -> PASS (out of gate scope).
  4. Pre-sentinel prereg, theory_grant=true, no escape field -> PASS (grandfathered).
  5. Sentinel-date prereg (2026-05-23 exact) -> BLOCK (sentinel is inclusive).
  6. theory_grant key absent entirely -> PASS.
  7. Escape field present but value=false -> BLOCK (only ``true`` clears).
  8. Multiple preregs in dir, mixed states -> only the offending ones returned.
  9. Tolerates ``True`` (capitalized) for the boolean parse.
 10. Empty hypotheses dir -> no violations.
 11. Continuity-grandfather escape field clears the gate.
 12. Continuity-grandfather escape with value=false does NOT clear the gate.
"""

from __future__ import annotations

import textwrap

from pipeline.check_drift import check_amendment_3_4_provisional_gate


def _write_prereg(
    hyp_dir,
    slug: str,
    *,
    theory_grant: bool | None = None,
    escape: bool | None = None,
    continuity: bool | None = None,
    extra: str = "",
) -> None:
    """Write a minimal prereg yaml. ``None`` omits the field entirely."""
    lines = ["metadata:"]
    if theory_grant is not None:
        lines.append(f"  theory_grant: {str(theory_grant).lower()}")
    if escape is not None:
        lines.append(f"  amendment_3_4_re_audit_closed: {str(escape).lower()}")
    if continuity is not None:
        lines.append(f"  amendment_3_4_continuity_grandfather: {str(continuity).lower()}")
    if extra:
        lines.append(extra)
    content = "\n".join(lines) + "\n"
    (hyp_dir / f"{slug}.yaml").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Case 1: post-sentinel + theory_grant=true + no escape -> BLOCK
# ---------------------------------------------------------------------------


def test_violation_post_sentinel_theory_true_no_escape(tmp_path):
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    _write_prereg(hyp_dir, "2026-05-24-some-new-cell-v1", theory_grant=True)

    violations = check_amendment_3_4_provisional_gate(hypotheses_dir=hyp_dir)
    assert violations, "Expected a violation for theory_grant=true without escape"
    combined = "\n".join(violations)
    assert "2026-05-24-some-new-cell-v1.yaml" in combined
    assert "Amendment 3.4" in combined
    assert "amendment_3_4_re_audit_closed" in combined


# ---------------------------------------------------------------------------
# Case 2: post-sentinel + theory_grant=true + escape=true -> PASS
# ---------------------------------------------------------------------------


def test_pass_post_sentinel_theory_true_with_escape(tmp_path):
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    _write_prereg(
        hyp_dir,
        "2026-06-01-cell-after-reaudit-v1",
        theory_grant=True,
        escape=True,
    )

    violations = check_amendment_3_4_provisional_gate(hypotheses_dir=hyp_dir)
    assert not violations, f"Expected no violations, got: {violations}"


# ---------------------------------------------------------------------------
# Case 3: post-sentinel + theory_grant=false -> PASS (out of scope)
# ---------------------------------------------------------------------------


def test_pass_post_sentinel_theory_false(tmp_path):
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    _write_prereg(hyp_dir, "2026-05-30-honest-no-theory-v1", theory_grant=False)

    violations = check_amendment_3_4_provisional_gate(hypotheses_dir=hyp_dir)
    assert not violations


# ---------------------------------------------------------------------------
# Case 4: pre-sentinel + theory_grant=true + no escape -> PASS (grandfathered)
# ---------------------------------------------------------------------------


def test_pass_pre_sentinel_grandfathered(tmp_path):
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    # 2026-05-22 is one day before the 2026-05-23 sentinel.
    _write_prereg(hyp_dir, "2026-05-22-pre-amendment-v1", theory_grant=True)

    violations = check_amendment_3_4_provisional_gate(hypotheses_dir=hyp_dir)
    assert not violations, "Pre-sentinel files must be grandfathered"


# ---------------------------------------------------------------------------
# Case 5: sentinel date 2026-05-23 itself -> BLOCK (inclusive)
# ---------------------------------------------------------------------------


def test_violation_sentinel_date_is_inclusive(tmp_path):
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    _write_prereg(hyp_dir, "2026-05-23-same-day-as-amendment-v1", theory_grant=True)

    violations = check_amendment_3_4_provisional_gate(hypotheses_dir=hyp_dir)
    assert violations, "Sentinel date is inclusive; must block same-day files"


# ---------------------------------------------------------------------------
# Case 6: theory_grant key absent -> PASS
# ---------------------------------------------------------------------------


def test_pass_theory_grant_key_absent(tmp_path):
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    # No theory_grant key at all (e.g. a template stub).
    (hyp_dir / "2026-05-30-template-stub-v1.yaml").write_text(
        "metadata:\n  name: stub\n", encoding="utf-8"
    )

    violations = check_amendment_3_4_provisional_gate(hypotheses_dir=hyp_dir)
    assert not violations


# ---------------------------------------------------------------------------
# Case 7: escape field present but value=false -> BLOCK
# ---------------------------------------------------------------------------


def test_violation_escape_field_false(tmp_path):
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    _write_prereg(
        hyp_dir,
        "2026-05-30-escape-false-v1",
        theory_grant=True,
        escape=False,
    )

    violations = check_amendment_3_4_provisional_gate(hypotheses_dir=hyp_dir)
    assert violations, "escape=false must NOT clear the gate"


# ---------------------------------------------------------------------------
# Case 8: mixed dir -> only offending entries returned
# ---------------------------------------------------------------------------


def test_mixed_dir_only_offenders_returned(tmp_path):
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    _write_prereg(hyp_dir, "2026-05-22-old-grandfathered-v1", theory_grant=True)
    _write_prereg(hyp_dir, "2026-05-30-clean-no-theory-v1", theory_grant=False)
    _write_prereg(hyp_dir, "2026-05-30-escaped-v1", theory_grant=True, escape=True)
    _write_prereg(hyp_dir, "2026-05-30-offender-v1", theory_grant=True)

    violations = check_amendment_3_4_provisional_gate(hypotheses_dir=hyp_dir)
    assert len(violations) == 1
    assert "2026-05-30-offender-v1.yaml" in violations[0]


# ---------------------------------------------------------------------------
# Case 9: capitalized ``True`` parses as truthy
# ---------------------------------------------------------------------------


def test_capitalized_true_parses(tmp_path):
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    (hyp_dir / "2026-05-30-cap-true-v1.yaml").write_text(
        textwrap.dedent(
            """\
            metadata:
              theory_grant: True
            """
        ),
        encoding="utf-8",
    )

    violations = check_amendment_3_4_provisional_gate(hypotheses_dir=hyp_dir)
    assert violations, "Capitalized True must still trigger the gate"


# ---------------------------------------------------------------------------
# Case 10: empty dir -> no violations
# ---------------------------------------------------------------------------


def test_empty_dir(tmp_path):
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    violations = check_amendment_3_4_provisional_gate(hypotheses_dir=hyp_dir)
    assert violations == []


# ---------------------------------------------------------------------------
# Mutation probe: confirm the regex actually inspects content, not filename
# ---------------------------------------------------------------------------


def test_mutation_probe_regex_inspects_content(tmp_path):
    """If someone refactors the check to skip content reading and pass on
    filename alone, this test fails. A post-sentinel filename with the
    correct theory_grant=true content but escape field absent must block.
    """
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    # Filename contains the words "amendment_3_4_re_audit_closed" but the
    # ACTUAL metadata does NOT — the check must read the file, not the name.
    (hyp_dir / "2026-05-30-amendment_3_4_re_audit_closed-decoy-v1.yaml").write_text(
        "metadata:\n  theory_grant: true\n", encoding="utf-8"
    )

    violations = check_amendment_3_4_provisional_gate(hypotheses_dir=hyp_dir)
    assert violations, "Filename containing the escape phrase must not satisfy the gate"


# ---------------------------------------------------------------------------
# Case 11: continuity-grandfather escape clears the gate
# ---------------------------------------------------------------------------


def test_pass_continuity_grandfather_clears_gate(tmp_path):
    """The three pre-existing PASS_PROTOCOL_A cells named in Amendment 3.4
    use the continuity-grandfather field instead of re-audit-closed.
    """
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    _write_prereg(
        hyp_dir,
        "2026-05-23-existing-cell-v1",
        theory_grant=True,
        continuity=True,
    )

    violations = check_amendment_3_4_provisional_gate(hypotheses_dir=hyp_dir)
    assert not violations, f"continuity-grandfather=true must clear the gate, got: {violations}"


# ---------------------------------------------------------------------------
# Case 12: continuity-grandfather=false does NOT clear the gate
# ---------------------------------------------------------------------------


def test_violation_continuity_grandfather_false(tmp_path):
    hyp_dir = tmp_path / "hypotheses"
    hyp_dir.mkdir()
    _write_prereg(
        hyp_dir,
        "2026-05-30-continuity-false-v1",
        theory_grant=True,
        continuity=False,
    )

    violations = check_amendment_3_4_provisional_gate(hypotheses_dir=hyp_dir)
    assert violations, "continuity-grandfather=false must NOT clear the gate"
