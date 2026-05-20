"""Injection tests for check_fast_lane_trial_ledger_append_only (Check #169).

Six mutation probes that flip a single invariant on a synthetic ledger file
and assert the check returns a violation naming the broken invariant:

  1. timestamp regression           -> append-only violation
  2. prior-entry mutation           -> structural_hash drift catch
  3. duplicate run_id               -> append-only violation
  4. malformed YAML / banner scrub  -> banner / format catch
  5. holdout_policy stripped        -> Mode A boundary breach
  6. holdout_sacred_from mutated    -> Mode A boundary breach

Plus a clean-state baseline test against a freshly-written ledger.

Also includes a writer-side capital-class refusal test: the
``append_trial_ledger_entry`` writer must REFUSE entries whose
``prereg_path`` names ``validated_setups`` / ``chordia_audit_log.yaml`` /
``lane_allocation.json`` / ``trading_app/live/`` (per design grounding
§ "Hard Constraints" + § "Files NOT to TOUCH").

Design grounding:
  docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from pipeline.check_drift import check_fast_lane_trial_ledger_append_only
from scripts.research.fast_lane_trial_ledger import (
    CapitalClassWriteRefused,
    HOLDOUT_POLICY_SENTINEL,
    HOLDOUT_SACRED_FROM_SENTINEL,
    LedgerAppendOnlyViolation,
    LedgerEntry,
    append_trial_ledger_entry,
)


# A canonical clean-state ledger fragment with two valid entries that share
# the boundary sentinels. Every injection probe starts from this and mutates
# exactly one field.
_VALID_LEDGER_TEXT = dedent(
    """\
    do_not_hand_edit: true
    schema_version: 1
    entries:
      - run_id: run-001
        run_timestamp_utc: '2026-05-20T01:00:00Z'
        prereg_path: docs/audit/hypotheses/2026-05-20-foo.yaml
        prereg_sha: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        structural_hash: deadbeefcafebabe
        template_version: fast_lane_v5.1
        testing_mode: individual
        pathway: A
        K_declared: 1
        holdout_policy: mode_A
        holdout_sacred_from: '2026-01-01'
        k_lineage: {}
        n_hat: null
        upstream_provenance: {}
        outcome: {}
      - run_id: run-002
        run_timestamp_utc: '2026-05-20T02:00:00Z'
        prereg_path: docs/audit/hypotheses/2026-05-20-bar.yaml
        prereg_sha: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
        structural_hash: 0123456789abcdef
        template_version: fast_lane_v5.1
        testing_mode: family
        pathway: A
        K_declared: 4
        holdout_policy: mode_A
        holdout_sacred_from: '2026-01-01'
        k_lineage: {}
        n_hat: null
        upstream_provenance: {}
        outcome: {}
    """
)


def _write_ledger(tmp_path: Path, text: str = _VALID_LEDGER_TEXT) -> Path:
    p = tmp_path / "fast_lane_trial_ledger.yaml"
    p.write_text(text, encoding="utf-8")
    return p


# ----------------------------------------------------------------------
# Clean-state baseline
# ----------------------------------------------------------------------


def test_clean_state_passes(tmp_path: Path):
    target = _write_ledger(tmp_path)
    assert check_fast_lane_trial_ledger_append_only(ledger_path=target) == []


def test_real_repo_ledger_passes():
    """The real on-disk ledger landed by Stage 2A.2 must pass clean."""
    assert check_fast_lane_trial_ledger_append_only() == []


# ----------------------------------------------------------------------
# Fail-closed: missing ledger file
# ----------------------------------------------------------------------


def test_missing_ledger_fails_closed(tmp_path: Path):
    forged = tmp_path / "does-not-exist.yaml"
    violations = check_fast_lane_trial_ledger_append_only(ledger_path=forged)
    assert violations
    assert any("ledger file missing" in v for v in violations)


# ----------------------------------------------------------------------
# Fail-closed: malformed YAML in the ledger file
# ----------------------------------------------------------------------


def test_malformed_ledger_yaml_fails_closed(tmp_path: Path):
    """check_fast_lane_trial_ledger_append_only must return a violation
    (not raise) when the ledger file contains unparseable YAML.
    Covers the universal malformed-YAML fail-closed gap from code review."""
    bad = tmp_path / "fast_lane_trial_ledger.yaml"
    bad.write_text(": not: valid: yaml: {\n", encoding="utf-8")
    violations = check_fast_lane_trial_ledger_append_only(ledger_path=bad)
    assert violations
    assert any(
        "parse" in v.lower() or "yaml" in v.lower() or "failed" in v.lower()
        for v in violations
    )


# ----------------------------------------------------------------------
# Timestamp normalisation: mixed Z / +00:00 suffix comparison
# ----------------------------------------------------------------------


def test_mixed_suffix_monotonic_passes(tmp_path: Path):
    """A ledger with Z on the first entry and +00:00 on the second (same
    instant expressed differently) must NOT trigger TIMESTAMP_REGRESSION.
    This is the regression test for the raw-string comparison bug."""
    mixed_ok = dedent(
        """\
        do_not_hand_edit: true
        schema_version: 1
        entries:
          - run_id: run-A
            run_timestamp_utc: '2026-05-20T01:00:00+00:00'
            prereg_path: docs/audit/hypotheses/2026-05-20-foo.yaml
            prereg_sha: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
            structural_hash: deadbeefcafebabe
            template_version: fast_lane_v5.1
            testing_mode: individual
            pathway: A
            K_declared: 1
            holdout_policy: mode_A
            holdout_sacred_from: '2026-01-01'
            k_lineage: {}
            n_hat: null
            upstream_provenance: {}
            outcome: {}
          - run_id: run-B
            run_timestamp_utc: '2026-05-20T02:00:00Z'
            prereg_path: docs/audit/hypotheses/2026-05-20-bar.yaml
            prereg_sha: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            structural_hash: 0123456789abcdef
            template_version: fast_lane_v5.1
            testing_mode: family
            pathway: A
            K_declared: 4
            holdout_policy: mode_A
            holdout_sacred_from: '2026-01-01'
            k_lineage: {}
            n_hat: null
            upstream_provenance: {}
            outcome: {}
        """
    )
    target = _write_ledger(tmp_path, mixed_ok)
    violations = check_fast_lane_trial_ledger_append_only(ledger_path=target)
    assert not any("TIMESTAMP_REGRESSION" in v for v in violations), (
        f"false TIMESTAMP_REGRESSION on mixed suffix pair: {violations}"
    )


def test_mixed_suffix_regression_is_caught(tmp_path: Path):
    """A ledger where the +00:00-suffix second entry is actually EARLIER
    than the Z-suffix first entry must trigger TIMESTAMP_REGRESSION."""
    mixed_bad = dedent(
        """\
        do_not_hand_edit: true
        schema_version: 1
        entries:
          - run_id: run-A
            run_timestamp_utc: '2026-05-20T03:00:00Z'
            prereg_path: docs/audit/hypotheses/2026-05-20-foo.yaml
            prereg_sha: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
            structural_hash: deadbeefcafebabe
            template_version: fast_lane_v5.1
            testing_mode: individual
            pathway: A
            K_declared: 1
            holdout_policy: mode_A
            holdout_sacred_from: '2026-01-01'
            k_lineage: {}
            n_hat: null
            upstream_provenance: {}
            outcome: {}
          - run_id: run-B
            run_timestamp_utc: '2026-05-20T01:00:00+00:00'
            prereg_path: docs/audit/hypotheses/2026-05-20-bar.yaml
            prereg_sha: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            structural_hash: 0123456789abcdef
            template_version: fast_lane_v5.1
            testing_mode: family
            pathway: A
            K_declared: 4
            holdout_policy: mode_A
            holdout_sacred_from: '2026-01-01'
            k_lineage: {}
            n_hat: null
            upstream_provenance: {}
            outcome: {}
        """
    )
    target = _write_ledger(tmp_path, mixed_bad)
    violations = check_fast_lane_trial_ledger_append_only(ledger_path=target)
    assert any("TIMESTAMP_REGRESSION" in v for v in violations), (
        f"expected TIMESTAMP_REGRESSION not in: {violations}"
    )


# ----------------------------------------------------------------------
# Injection 1: timestamp regression
# ----------------------------------------------------------------------


def test_timestamp_regression_is_caught(tmp_path: Path):
    mutated = _VALID_LEDGER_TEXT.replace(
        "run_timestamp_utc: '2026-05-20T02:00:00Z'",
        "run_timestamp_utc: '2026-05-19T00:00:00Z'",
    )
    target = _write_ledger(tmp_path, mutated)
    violations = check_fast_lane_trial_ledger_append_only(ledger_path=target)
    assert violations
    assert any("TIMESTAMP_REGRESSION" in v for v in violations)


# ----------------------------------------------------------------------
# Injection 2: prior-entry mutation (structural_hash rewrite is fail-closed
# via the format check; this probe specifically rewrites the prior entry's
# structural_hash to one that is the wrong length)
# ----------------------------------------------------------------------


def test_prior_entry_structural_hash_mutation_is_caught(tmp_path: Path):
    mutated = _VALID_LEDGER_TEXT.replace(
        "structural_hash: deadbeefcafebabe",
        "structural_hash: tampered",
    )
    target = _write_ledger(tmp_path, mutated)
    violations = check_fast_lane_trial_ledger_append_only(ledger_path=target)
    assert violations
    assert any("STRUCTURAL_HASH_LENGTH" in v or "STRUCTURAL_HASH_NOT_HEX" in v for v in violations)


# ----------------------------------------------------------------------
# Injection 3: duplicate run_id
# ----------------------------------------------------------------------


def test_duplicate_run_id_is_caught(tmp_path: Path):
    mutated = _VALID_LEDGER_TEXT.replace(
        "run_id: run-002",
        "run_id: run-001",
    )
    target = _write_ledger(tmp_path, mutated)
    violations = check_fast_lane_trial_ledger_append_only(ledger_path=target)
    assert violations
    assert any("DUPLICATE run_id" in v for v in violations)


# ----------------------------------------------------------------------
# Injection 4: banner scrub (do_not_hand_edit removed)
# ----------------------------------------------------------------------


def test_banner_scrub_is_caught(tmp_path: Path):
    mutated = _VALID_LEDGER_TEXT.replace("do_not_hand_edit: true\n", "")
    target = _write_ledger(tmp_path, mutated)
    violations = check_fast_lane_trial_ledger_append_only(ledger_path=target)
    assert violations
    assert any("BANNER TAMPERED" in v for v in violations)


# ----------------------------------------------------------------------
# Injection 5: holdout_policy stripped
# ----------------------------------------------------------------------


def test_holdout_policy_stripped_is_caught(tmp_path: Path):
    # Drop holdout_policy from both entries (line-level removal); the check
    # must flag the missing required field on at least one entry.
    mutated = (
        "\n".join(line for line in _VALID_LEDGER_TEXT.splitlines() if not line.lstrip().startswith("holdout_policy:"))
        + "\n"
    )
    target = _write_ledger(tmp_path, mutated)
    violations = check_fast_lane_trial_ledger_append_only(ledger_path=target)
    assert violations
    assert any("HOLDOUT_SENTINEL" in v or "missing required field 'holdout_policy'" in v for v in violations)


# ----------------------------------------------------------------------
# Injection 6: holdout_sacred_from mutated
# ----------------------------------------------------------------------


def test_holdout_sacred_from_mutated_is_caught(tmp_path: Path):
    mutated = _VALID_LEDGER_TEXT.replace(
        "holdout_sacred_from: '2026-01-01'",
        "holdout_sacred_from: '2025-01-01'",
    )
    target = _write_ledger(tmp_path, mutated)
    violations = check_fast_lane_trial_ledger_append_only(ledger_path=target)
    assert violations
    assert any("HOLDOUT_SENTINEL" in v for v in violations)


# ----------------------------------------------------------------------
# Capital-class writer refusal (one test per forbidden substring)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden_path",
    [
        "validated_setups/abc.yaml",
        "docs/runtime/chordia_audit_log.yaml",
        "docs/runtime/lane_allocation.json",
        "trading_app/live/orchestrator.py",
        "trading_app\\live\\bot_state.py",
    ],
)
def test_writer_refuses_capital_class_paths(tmp_path: Path, forbidden_path: str):
    """The ledger writer MUST refuse to record entries whose prereg_path
    crosses the capital-class boundary (CLAUDE.md § Source-of-Truth Chain
    Rule + Stage 2A design grounding § Hard Constraints)."""
    target = _write_ledger(tmp_path)
    entry = LedgerEntry(
        run_id="run-999",
        run_timestamp_utc="2026-05-20T03:00:00Z",
        prereg_path=forbidden_path,
        prereg_sha="cccccccccccccccccccccccccccccccccccccccc",
        structural_hash="fedcba9876543210",
        template_version="fast_lane_v5.1",
        testing_mode="individual",
        pathway="A",
        K_declared=1,
    )
    with pytest.raises(CapitalClassWriteRefused):
        append_trial_ledger_entry(target, entry)


def test_writer_refuses_duplicate_run_id_runtime(tmp_path: Path):
    """The writer must also refuse a duplicate run_id at append time, not
    only via the static check."""
    target = _write_ledger(tmp_path)
    dup = LedgerEntry(
        run_id="run-001",  # already in the seed ledger
        run_timestamp_utc="2026-05-20T03:00:00Z",
        prereg_path="docs/audit/hypotheses/2026-05-20-baz.yaml",
        prereg_sha="dddddddddddddddddddddddddddddddddddddddd",
        structural_hash="aaaaaaaaaaaaaaaa",
        template_version="fast_lane_v5.1",
        testing_mode="individual",
        pathway="A",
        K_declared=1,
        holdout_policy=HOLDOUT_POLICY_SENTINEL,
        holdout_sacred_from=HOLDOUT_SACRED_FROM_SENTINEL,
    )
    with pytest.raises(LedgerAppendOnlyViolation):
        append_trial_ledger_entry(target, dup)


def test_writer_refuses_backwards_timestamp_runtime(tmp_path: Path):
    """The writer must refuse a non-monotonic incoming timestamp."""
    target = _write_ledger(tmp_path)
    backwards = LedgerEntry(
        run_id="run-003",
        run_timestamp_utc="2026-05-19T00:00:00Z",  # before last entry
        prereg_path="docs/audit/hypotheses/2026-05-20-qux.yaml",
        prereg_sha="eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        structural_hash="bbbbbbbbbbbbbbbb",
        template_version="fast_lane_v5.1",
        testing_mode="individual",
        pathway="A",
        K_declared=1,
    )
    with pytest.raises(LedgerAppendOnlyViolation):
        append_trial_ledger_entry(target, backwards)
