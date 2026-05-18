"""Mutation-probe + sibling-coverage tests for the canonical-inline-copies registry.

Covers:
  * Registry integrity (uniqueness, validation, canonical_class allowlist).
  * Meta-check ``check_canonical_inline_copies_have_parity_check`` —
    clean-state pass + ≥5 negative mutation probes.
  * Per-entry parity-check tests, one test function per gated constant
    (sibling-coverage doctrine — see
    ``memory/feedback_regex_alternation_sibling_coverage.md``).

The slug-pattern naming convention is mandatory: meta-check counts
``def test_*<slug>*`` functions and requires at least
``len(entry.gated_constants)`` per entry. Renaming a test silently breaks
the sibling-coverage guarantee.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from pipeline import canonical_inline_copies as registry_mod
from pipeline.canonical_inline_copies import (
    CANONICAL_INLINE_COPIES,
    InlineCopyPair,
)
from pipeline import check_drift


# ---------------------------------------------------------------------------
# Registry-integrity tests (validator runs at import time; these double-cover).
# ---------------------------------------------------------------------------


def test_registry_has_at_least_seed_entries() -> None:
    assert len(CANONICAL_INLINE_COPIES) >= 3, (
        "Stage 2 seed list lost entries; expected at minimum 3 verified pairs"
    )


def test_registry_slug_uniqueness() -> None:
    slugs = [e.slug for e in CANONICAL_INLINE_COPIES]
    assert len(slugs) == len(set(slugs)), f"duplicate slugs in registry: {slugs}"


def test_registry_parity_func_uniqueness() -> None:
    funcs = [e.parity_check_func for e in CANONICAL_INLINE_COPIES]
    assert len(funcs) == len(set(funcs)), (
        f"two registry entries share a parity check (meta-check would be blind): {funcs}"
    )


def test_registry_validator_rejects_duplicate_slug() -> None:
    bad = [
        CANONICAL_INLINE_COPIES[0],
        InlineCopyPair(
            slug=CANONICAL_INLINE_COPIES[0].slug,
            inline_site="x:1",
            canonical_source="y:1",
            gated_constants=("A",),
            parity_check_func="different_name",
            test_slug="t",
            rationale="dup",
        ),
    ]
    with pytest.raises(ValueError, match="duplicate slug"):
        registry_mod._validate_registry(bad)


def test_registry_validator_rejects_duplicate_parity_func() -> None:
    first = CANONICAL_INLINE_COPIES[0]
    bad = [
        first,
        InlineCopyPair(
            slug="different_slug_for_test",
            inline_site="x:1",
            canonical_source="y:1",
            gated_constants=("A",),
            parity_check_func=first.parity_check_func,
            test_slug="t",
            rationale="dup",
        ),
    ]
    with pytest.raises(ValueError, match="duplicate parity_check_func"):
        registry_mod._validate_registry(bad)


def test_registry_validator_rejects_empty_gated_constants() -> None:
    bad = [
        InlineCopyPair(
            slug="empty_test",
            inline_site="x:1",
            canonical_source="y:1",
            gated_constants=(),
            parity_check_func="check_empty",
            test_slug="t",
            rationale="empty",
        ),
    ]
    with pytest.raises(ValueError, match="empty gated_constants"):
        registry_mod._validate_registry(bad)


def test_registry_validator_rejects_unknown_canonical_class() -> None:
    bad = [
        InlineCopyPair(
            slug="unknown_class_test",
            inline_site="x:1",
            canonical_source="y:1",
            gated_constants=("A",),
            parity_check_func="check_unknown",
            test_slug="t",
            rationale="bad class",
            canonical_class="banana",
        ),
    ]
    with pytest.raises(ValueError, match="unknown canonical_class"):
        registry_mod._validate_registry(bad)


# ---------------------------------------------------------------------------
# Meta-check tests: clean state + mutation probes.
# ---------------------------------------------------------------------------


def test_meta_check_clean_state_passes() -> None:
    """All registry entries currently satisfy the meta-check."""
    violations = check_drift.check_canonical_inline_copies_have_parity_check()
    assert violations == [], f"meta-check unexpected violations: {violations}"


def test_meta_check_detects_missing_parity_function() -> None:
    """Drift probe (a): entry points to nonexistent function."""
    bogus = [
        InlineCopyPair(
            slug="probe_missing_func",
            inline_site="x:1",
            canonical_source="y:1",
            gated_constants=("CONST_A",),
            parity_check_func="check_function_that_does_not_exist_anywhere",
            test_slug="test_canonical_inline_copies_registry",
            rationale="probe",
        ),
    ]
    with mock.patch.object(
        check_drift,
        "check_canonical_inline_copies_have_parity_check",
        wraps=check_drift.check_canonical_inline_copies_have_parity_check,
    ):
        with mock.patch(
            "pipeline.canonical_inline_copies.CANONICAL_INLINE_COPIES",
            bogus,
        ):
            violations = check_drift.check_canonical_inline_copies_have_parity_check()
    assert any(
        "check_function_that_does_not_exist_anywhere" in v for v in violations
    ), f"expected missing-function violation, got: {violations}"


def test_meta_check_detects_noncallable_parity_attribute() -> None:
    """Drift probe (b): entry points to a non-callable name."""
    # Plant a non-callable in module globals, then point a fake entry at it.
    sentinel_name = "_test_planted_noncallable_42"
    check_drift_globals = vars(check_drift)
    check_drift_globals[sentinel_name] = 42  # int — not callable
    try:
        bogus = [
            InlineCopyPair(
                slug="probe_noncallable",
                inline_site="x:1",
                canonical_source="y:1",
                gated_constants=("CONST_A",),
                parity_check_func=sentinel_name,
                test_slug="test_canonical_inline_copies_registry",
                rationale="probe",
            ),
        ]
        with mock.patch(
            "pipeline.canonical_inline_copies.CANONICAL_INLINE_COPIES",
            bogus,
        ):
            violations = check_drift.check_canonical_inline_copies_have_parity_check()
        assert any("non-callable" in v for v in violations), (
            f"expected non-callable violation, got: {violations}"
        )
    finally:
        del check_drift_globals[sentinel_name]


def test_meta_check_detects_missing_test_file() -> None:
    """Drift probe (c): entry points to a nonexistent test file."""
    bogus = [
        InlineCopyPair(
            slug="probe_missing_test_file",
            inline_site="x:1",
            canonical_source="y:1",
            gated_constants=("CONST_A",),
            parity_check_func="check_canonical_inline_copies_have_parity_check",  # exists
            test_slug="test_file_that_does_not_exist_anywhere_in_test_pipeline",
            rationale="probe",
        ),
    ]
    with mock.patch(
        "pipeline.canonical_inline_copies.CANONICAL_INLINE_COPIES",
        bogus,
    ):
        violations = check_drift.check_canonical_inline_copies_have_parity_check()
    assert any(
        "test_file_that_does_not_exist_anywhere_in_test_pipeline" in v
        for v in violations
    ), f"expected missing-test-file violation, got: {violations}"


def test_meta_check_detects_insufficient_test_count() -> None:
    """Drift probe (d): test file exists but lacks sibling coverage for N constants."""
    # Pretend an entry claims 99 gated constants — current test file has nowhere
    # near 99 slug-matched tests so the sibling-coverage rule must fire.
    bogus = [
        InlineCopyPair(
            slug="probe_insufficient_tests",
            inline_site="x:1",
            canonical_source="y:1",
            gated_constants=tuple(f"K{i}" for i in range(99)),
            parity_check_func="check_canonical_inline_copies_have_parity_check",
            test_slug="test_canonical_inline_copies_registry",
            rationale="probe",
        ),
    ]
    with mock.patch(
        "pipeline.canonical_inline_copies.CANONICAL_INLINE_COPIES",
        bogus,
    ):
        violations = check_drift.check_canonical_inline_copies_have_parity_check()
    assert any("sibling-coverage doctrine requires" in v for v in violations), (
        f"expected sibling-coverage violation, got: {violations}"
    )


def test_meta_check_fail_closed_on_registry_import_error() -> None:
    """If the registry module cannot import, return a violation — not a crash."""
    with mock.patch.dict(
        "sys.modules",
        {"pipeline.canonical_inline_copies": None},
    ):
        violations = check_drift.check_canonical_inline_copies_have_parity_check()
    assert any("failed to import registry" in v for v in violations), (
        f"expected import-failure violation, got: {violations}"
    )


# ---------------------------------------------------------------------------
# c8_fail_labels: 6 gated constants → 6 slug-matching test functions.
# ---------------------------------------------------------------------------


def test_c8_fail_labels_clean_state_passes() -> None:
    violations = check_drift.check_c8_fail_labels_parity()
    assert violations == [], f"c8_fail_labels parity unexpectedly violated: {violations}"


def test_c8_fail_labels_detects_only_in_drift(tmp_path: Path, monkeypatch) -> None:
    """Drift in check_drift.py adding a label not in lane_allocator must violate."""
    # Mutate the check by monkeypatching PROJECT_ROOT to a fake tree with
    # divergent files.
    drift_dir = tmp_path / "pipeline"
    drift_dir.mkdir()
    alloc_dir = tmp_path / "trading_app"
    alloc_dir.mkdir()
    (drift_dir / "check_drift.py").write_text(
        '# Mirrors apply_c8_gate._C8_FAIL_LABELS\n'
        'c8_fail_labels = {"FAILED_RATIO", "NEGATIVE_OOS_EXPR", "EXTRA_LABEL_ONLY_IN_DRIFT"}\n',
        encoding="utf-8",
    )
    (alloc_dir / "lane_allocator.py").write_text(
        '# Set of c8 labels that mean "not deployable" per Criterion 8 + Amendment 3.1.\n'
        '_C8_FAIL_LABELS = {"FAILED_RATIO", "NEGATIVE_OOS_EXPR"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
    violations = check_drift.check_c8_fail_labels_parity()
    assert any("EXTRA_LABEL_ONLY_IN_DRIFT" in v for v in violations), violations


def test_c8_fail_labels_detects_only_in_lane_allocator(tmp_path: Path, monkeypatch) -> None:
    """Allocator adds a label drift check doesn't know about (fail-open class)."""
    drift_dir = tmp_path / "pipeline"
    drift_dir.mkdir()
    alloc_dir = tmp_path / "trading_app"
    alloc_dir.mkdir()
    (drift_dir / "check_drift.py").write_text(
        '# Mirrors apply_c8_gate._C8_FAIL_LABELS\n'
        'c8_fail_labels = {"FAILED_RATIO"}\n',
        encoding="utf-8",
    )
    (alloc_dir / "lane_allocator.py").write_text(
        '# Set of c8 labels that mean "not deployable" per Criterion 8 + Amendment 3.1.\n'
        '_C8_FAIL_LABELS = {"FAILED_RATIO", "NEW_HIDDEN_FAIL_LABEL"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
    violations = check_drift.check_c8_fail_labels_parity()
    assert any("NEW_HIDDEN_FAIL_LABEL" in v for v in violations), violations
    assert any("fail-open" in v for v in violations), violations


def test_c8_fail_labels_detects_missing_anchor_comment(tmp_path: Path, monkeypatch) -> None:
    """If the anchor comment is renamed, the check must fail-closed (not silently pass)."""
    drift_dir = tmp_path / "pipeline"
    drift_dir.mkdir()
    alloc_dir = tmp_path / "trading_app"
    alloc_dir.mkdir()
    (drift_dir / "check_drift.py").write_text(
        '# completely unrelated comment\n'
        'c8_fail_labels = {"FAILED_RATIO"}\n',
        encoding="utf-8",
    )
    (alloc_dir / "lane_allocator.py").write_text(
        '# Set of c8 labels that mean "not deployable" per Criterion 8 + Amendment 3.1.\n'
        '_C8_FAIL_LABELS = {"FAILED_RATIO"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
    violations = check_drift.check_c8_fail_labels_parity()
    assert any("anchor" in v.lower() for v in violations), violations


def test_c8_fail_labels_failed_ratio_individually_present() -> None:
    """Sibling-coverage: per gated constant FAILED_RATIO present in both sets."""
    import importlib

    cd = importlib.import_module("pipeline.check_drift")
    text = (cd.PROJECT_ROOT / "pipeline" / "check_drift.py").read_text(encoding="utf-8")
    assert '"FAILED_RATIO"' in text
    alloc_text = (
        cd.PROJECT_ROOT / "trading_app" / "lane_allocator.py"
    ).read_text(encoding="utf-8")
    assert '"FAILED_RATIO"' in alloc_text


def test_c8_fail_labels_negative_oos_expr_individually_present() -> None:
    """Sibling-coverage: NEGATIVE_OOS_EXPR present in both sets."""
    cd_text = (
        check_drift.PROJECT_ROOT / "pipeline" / "check_drift.py"
    ).read_text(encoding="utf-8")
    alloc_text = (
        check_drift.PROJECT_ROOT / "trading_app" / "lane_allocator.py"
    ).read_text(encoding="utf-8")
    assert '"NEGATIVE_OOS_EXPR"' in cd_text
    assert '"NEGATIVE_OOS_EXPR"' in alloc_text


def test_c8_fail_labels_no_oos_data_individually_present() -> None:
    """Sibling-coverage: NO_OOS_DATA present in both sets."""
    cd_text = (
        check_drift.PROJECT_ROOT / "pipeline" / "check_drift.py"
    ).read_text(encoding="utf-8")
    alloc_text = (
        check_drift.PROJECT_ROOT / "trading_app" / "lane_allocator.py"
    ).read_text(encoding="utf-8")
    assert '"NO_OOS_DATA"' in cd_text
    assert '"NO_OOS_DATA"' in alloc_text


def test_c8_fail_labels_insufficient_n_pathway_b_individually_present() -> None:
    """Sibling-coverage: INSUFFICIENT_N_PATHWAY_B_REJECT present in both sets."""
    cd_text = (
        check_drift.PROJECT_ROOT / "pipeline" / "check_drift.py"
    ).read_text(encoding="utf-8")
    alloc_text = (
        check_drift.PROJECT_ROOT / "trading_app" / "lane_allocator.py"
    ).read_text(encoding="utf-8")
    assert '"INSUFFICIENT_N_PATHWAY_B_REJECT"' in cd_text
    assert '"INSUFFICIENT_N_PATHWAY_B_REJECT"' in alloc_text


def test_c8_fail_labels_insufficient_n_pathway_a_individually_present() -> None:
    """Sibling-coverage: INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH present in both sets."""
    cd_text = (
        check_drift.PROJECT_ROOT / "pipeline" / "check_drift.py"
    ).read_text(encoding="utf-8")
    alloc_text = (
        check_drift.PROJECT_ROOT / "trading_app" / "lane_allocator.py"
    ).read_text(encoding="utf-8")
    assert '"INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH"' in cd_text
    assert '"INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH"' in alloc_text


def test_c8_fail_labels_rejected_individually_present() -> None:
    """Sibling-coverage: REJECTED present in both sets."""
    cd_text = (
        check_drift.PROJECT_ROOT / "pipeline" / "check_drift.py"
    ).read_text(encoding="utf-8")
    alloc_text = (
        check_drift.PROJECT_ROOT / "trading_app" / "lane_allocator.py"
    ).read_text(encoding="utf-8")
    assert '"REJECTED"' in cd_text
    assert '"REJECTED"' in alloc_text


# ---------------------------------------------------------------------------
# calibrate_null_sigmas: 3 gated constants → 3 slug-matching test functions.
# ---------------------------------------------------------------------------


def test_calibrate_null_sigmas_clean_state_passes() -> None:
    violations = check_drift.check_calibrate_null_sigmas_parity()
    assert violations == [], f"sigma parity unexpectedly violated: {violations}"


def test_calibrate_null_sigmas_detects_value_divergence(tmp_path: Path, monkeypatch) -> None:
    cal_dir = tmp_path / "scripts" / "tools"
    src_dir = tmp_path / "scripts" / "tests"
    cal_dir.mkdir(parents=True)
    src_dir.mkdir(parents=True)
    (cal_dir / "calibrate_null_sigma.py").write_text(
        'CURRENT_SIGMAS: dict[str, float] = {\n'
        '    "MGC": 1.5,\n'   # divergent
        '    "MNQ": 5.0,\n'
        '    "MES": 1.1,\n'
        '}\n',
        encoding="utf-8",
    )
    (src_dir / "run_null_batch.py").write_text(
        'INSTRUMENT_NULL_PARAMS: dict[str, dict] = {\n'
        '    "MGC": {"sigma": 1.2, "start_price": 2000.0, "tick_size": 0.10},\n'
        '    "MNQ": {"sigma": 5.0, "start_price": 20000.0, "tick_size": 0.25},\n'
        '    "MES": {"sigma": 1.1, "start_price": 5000.0, "tick_size": 0.25},\n'
        '}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
    violations = check_drift.check_calibrate_null_sigmas_parity()
    assert any("MGC" in v and "1.5" in v and "1.2" in v for v in violations), violations


def test_calibrate_null_sigmas_detects_missing_instrument_key(tmp_path: Path, monkeypatch) -> None:
    cal_dir = tmp_path / "scripts" / "tools"
    src_dir = tmp_path / "scripts" / "tests"
    cal_dir.mkdir(parents=True)
    src_dir.mkdir(parents=True)
    (cal_dir / "calibrate_null_sigma.py").write_text(
        'CURRENT_SIGMAS: dict[str, float] = {\n'
        '    "MGC": 1.2,\n'
        '    "MNQ": 5.0,\n'
        # MES missing
        '}\n',
        encoding="utf-8",
    )
    (src_dir / "run_null_batch.py").write_text(
        'INSTRUMENT_NULL_PARAMS: dict[str, dict] = {\n'
        '    "MGC": {"sigma": 1.2, "start_price": 2000.0, "tick_size": 0.10},\n'
        '    "MNQ": {"sigma": 5.0, "start_price": 20000.0, "tick_size": 0.25},\n'
        '    "MES": {"sigma": 1.1, "start_price": 5000.0, "tick_size": 0.25},\n'
        '}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
    violations = check_drift.check_calibrate_null_sigmas_parity()
    assert any("MES" in v and "missing" in v for v in violations), violations


def test_calibrate_null_sigmas_MGC_individually_in_sync() -> None:
    """Sibling-coverage: MGC sigma matches across producer + consumer."""
    cal_text = (
        check_drift.PROJECT_ROOT / "scripts" / "tools" / "calibrate_null_sigma.py"
    ).read_text(encoding="utf-8")
    src_text = (
        check_drift.PROJECT_ROOT / "scripts" / "tests" / "run_null_batch.py"
    ).read_text(encoding="utf-8")
    assert '"MGC": 1.2' in cal_text
    assert '"MGC": {"sigma": 1.2' in src_text


def test_calibrate_null_sigmas_MNQ_individually_in_sync() -> None:
    """Sibling-coverage: MNQ sigma matches across producer + consumer."""
    cal_text = (
        check_drift.PROJECT_ROOT / "scripts" / "tools" / "calibrate_null_sigma.py"
    ).read_text(encoding="utf-8")
    src_text = (
        check_drift.PROJECT_ROOT / "scripts" / "tests" / "run_null_batch.py"
    ).read_text(encoding="utf-8")
    assert '"MNQ": 5.0' in cal_text
    assert '"MNQ": {"sigma": 5.0' in src_text


def test_calibrate_null_sigmas_MES_individually_in_sync() -> None:
    """Sibling-coverage: MES sigma matches across producer + consumer."""
    cal_text = (
        check_drift.PROJECT_ROOT / "scripts" / "tools" / "calibrate_null_sigma.py"
    ).read_text(encoding="utf-8")
    src_text = (
        check_drift.PROJECT_ROOT / "scripts" / "tests" / "run_null_batch.py"
    ).read_text(encoding="utf-8")
    assert '"MES": 1.1' in cal_text
    assert '"MES": {"sigma": 1.1' in src_text


# ---------------------------------------------------------------------------
# criterion_ladder_chordia_thresholds: 2 gated constants → 2 slug-matching tests.
# ---------------------------------------------------------------------------


def test_criterion_ladder_chordia_thresholds_clean_state_passes() -> None:
    violations = check_drift.check_criterion_ladder_chordia_thresholds_parity()
    assert violations == [], f"Chordia ladder parity unexpectedly violated: {violations}"


def test_criterion_ladder_chordia_thresholds_detects_with_theory_drift(
    tmp_path: Path,
) -> None:
    """Inline T_THRESHOLD_WITH_THEORY drifts from canonical 3.00."""
    # Build a fake criteria doc with canonical row + inline file with drifted value
    fake_criteria = tmp_path / "pre_registered_criteria.md"
    fake_criteria.write_text(
        "## Acceptance matrix\n"
        "| 4 Chordia t-stat | t ≥ 3.00 (with theory) or 3.79 (without) | YES |\n",
        encoding="utf-8",
    )
    # Mutate inline file at expected location relative to a fake PROJECT_ROOT
    fake_ladder = tmp_path / "scripts" / "tools"
    fake_ladder.mkdir(parents=True)
    (fake_ladder / "criterion_ladder_check.py").write_text(
        "T_THRESHOLD_WITH_THEORY = 2.50\n"  # drifted
        "T_THRESHOLD_NO_THEORY = 3.79\n",
        encoding="utf-8",
    )
    # Re-route PROJECT_ROOT — pass criteria_path explicitly.
    with mock.patch.object(check_drift, "PROJECT_ROOT", tmp_path):
        violations = check_drift.check_criterion_ladder_chordia_thresholds_parity(
            criteria_path=fake_criteria,
        )
    assert any(
        "T_THRESHOLD_WITH_THEORY" in v and "2.5" in v for v in violations
    ), violations


def test_criterion_ladder_chordia_thresholds_detects_no_theory_drift(
    tmp_path: Path,
) -> None:
    fake_criteria = tmp_path / "pre_registered_criteria.md"
    fake_criteria.write_text(
        "| 4 Chordia t-stat | t ≥ 3.00 (with theory) or 3.79 (without) | YES |\n",
        encoding="utf-8",
    )
    fake_ladder = tmp_path / "scripts" / "tools"
    fake_ladder.mkdir(parents=True)
    (fake_ladder / "criterion_ladder_check.py").write_text(
        "T_THRESHOLD_WITH_THEORY = 3.00\n"
        "T_THRESHOLD_NO_THEORY = 4.00\n",  # drifted
        encoding="utf-8",
    )
    with mock.patch.object(check_drift, "PROJECT_ROOT", tmp_path):
        violations = check_drift.check_criterion_ladder_chordia_thresholds_parity(
            criteria_path=fake_criteria,
        )
    assert any(
        "T_THRESHOLD_NO_THEORY" in v and "4.0" in v for v in violations
    ), violations


def test_criterion_ladder_chordia_thresholds_doctrine_amendment_propagates(
    tmp_path: Path,
) -> None:
    """If the canonical doc is amended (e.g., 3.00 → 3.10), the check parses it.

    This is the doctrine-supersession-layer-trap guard
    (see feedback_chordia_threshold_doctrine_supersession_layer_trap.md):
    a hardcoded literal would fail here even when inline matches the new
    canonical. Our check reads canonical at runtime, so a coordinated
    amendment + inline update passes; only DRIFT fails.
    """
    fake_criteria = tmp_path / "pre_registered_criteria.md"
    fake_criteria.write_text(
        "| 4 Chordia t-stat | t ≥ 3.10 (with theory) or 3.79 (without) | YES |\n",
        encoding="utf-8",
    )
    fake_ladder = tmp_path / "scripts" / "tools"
    fake_ladder.mkdir(parents=True)
    (fake_ladder / "criterion_ladder_check.py").write_text(
        "T_THRESHOLD_WITH_THEORY = 3.10\n"  # matches amended canonical
        "T_THRESHOLD_NO_THEORY = 3.79\n",
        encoding="utf-8",
    )
    with mock.patch.object(check_drift, "PROJECT_ROOT", tmp_path):
        violations = check_drift.check_criterion_ladder_chordia_thresholds_parity(
            criteria_path=fake_criteria,
        )
    assert violations == [], (
        f"coordinated amendment + inline update must pass; got: {violations}"
    )


def test_criterion_ladder_chordia_thresholds_fail_closed_on_missing_canonical_row(
    tmp_path: Path,
) -> None:
    """If the acceptance-matrix row is missing/refactored, fail-closed (not silent pass)."""
    fake_criteria = tmp_path / "pre_registered_criteria.md"
    fake_criteria.write_text("# doc with no Criterion 4 row\n", encoding="utf-8")
    fake_ladder = tmp_path / "scripts" / "tools"
    fake_ladder.mkdir(parents=True)
    (fake_ladder / "criterion_ladder_check.py").write_text(
        "T_THRESHOLD_WITH_THEORY = 3.00\n"
        "T_THRESHOLD_NO_THEORY = 3.79\n",
        encoding="utf-8",
    )
    with mock.patch.object(check_drift, "PROJECT_ROOT", tmp_path):
        violations = check_drift.check_criterion_ladder_chordia_thresholds_parity(
            criteria_path=fake_criteria,
        )
    assert any("not found" in v.lower() or "drifted" in v.lower() for v in violations), violations


def test_criterion_ladder_chordia_thresholds_T_THRESHOLD_WITH_THEORY_individually_present() -> None:
    """Sibling-coverage: WITH_THEORY constant declaration present at the inline site."""
    ladder_text = (
        check_drift.PROJECT_ROOT / "scripts" / "tools" / "criterion_ladder_check.py"
    ).read_text(encoding="utf-8")
    assert "T_THRESHOLD_WITH_THEORY" in ladder_text


def test_criterion_ladder_chordia_thresholds_T_THRESHOLD_NO_THEORY_individually_present() -> None:
    """Sibling-coverage: NO_THEORY constant declaration present at the inline site."""
    ladder_text = (
        check_drift.PROJECT_ROOT / "scripts" / "tools" / "criterion_ladder_check.py"
    ).read_text(encoding="utf-8")
    assert "T_THRESHOLD_NO_THEORY" in ladder_text
