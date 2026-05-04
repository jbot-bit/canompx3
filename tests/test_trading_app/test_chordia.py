"""Tests for trading_app/chordia.py — Criterion 4 t-statistic gate.

Covers:
- compute_chordia_t identity ``t = sharpe * sqrt(N)`` against hand-computed values
- chordia_threshold returns the locked thresholds
- chordia_gate verdict at exact boundary, above, below
- N < 2 raises ValueError
- Three synthetic distributions: high-variance, low-variance, noise
- Theory vs no-theory threshold differs
- ChordiaAuditEntry addendum-field round-trip (PR #221 schema close)
"""

from __future__ import annotations

import logging
import math
from datetime import date

import pytest

from trading_app.chordia import (
    CHORDIA_T_WITH_THEORY,
    CHORDIA_T_WITHOUT_THEORY,
    chordia_gate,
    chordia_threshold,
    compute_chordia_t,
    load_chordia_audit_log,
)


class TestChordiaThreshold:
    """The locked threshold lookup."""

    def test_with_theory_returns_3_00(self):
        assert chordia_threshold(has_theory=True) == 3.00

    def test_without_theory_returns_3_79(self):
        assert chordia_threshold(has_theory=False) == 3.79

    def test_threshold_constants_match(self):
        # Sanity check the module-level constants are not accidentally rebound.
        assert CHORDIA_T_WITH_THEORY == 3.00
        assert CHORDIA_T_WITHOUT_THEORY == 3.79


class TestComputeChordiaT:
    """The t-statistic identity ``t = sharpe * sqrt(N)``."""

    def test_identity_at_n_100(self):
        # sharpe=0.30, N=100 → t = 0.30 * 10 = 3.0
        assert compute_chordia_t(0.30, 100) == pytest.approx(3.0, abs=1e-9)

    def test_identity_at_n_64(self):
        # sharpe=0.50, N=64 → t = 0.50 * 8 = 4.0
        assert compute_chordia_t(0.50, 64) == pytest.approx(4.0, abs=1e-9)

    def test_identity_at_n_400(self):
        # sharpe=0.20, N=400 → t = 0.20 * 20 = 4.0
        assert compute_chordia_t(0.20, 400) == pytest.approx(4.0, abs=1e-9)

    def test_negative_sharpe_negative_t(self):
        # A losing strategy produces a negative t.
        assert compute_chordia_t(-0.30, 100) == pytest.approx(-3.0, abs=1e-9)

    def test_zero_sharpe_zero_t(self):
        assert compute_chordia_t(0.0, 100) == 0.0

    def test_sample_size_below_2_raises(self):
        with pytest.raises(ValueError, match="sample_size >= 2"):
            compute_chordia_t(0.5, 1)

    def test_sample_size_zero_raises(self):
        with pytest.raises(ValueError, match="sample_size >= 2"):
            compute_chordia_t(0.5, 0)

    def test_sample_size_2_minimum_works(self):
        # N=2 is the smallest valid sample.
        result = compute_chordia_t(1.0, 2)
        assert result == pytest.approx(math.sqrt(2), abs=1e-9)


class TestChordiaGate:
    """End-to-end gate verdict + boundary behavior."""

    def test_with_theory_at_exact_boundary_passes(self):
        # sharpe=0.30, N=100 → t=3.0 == threshold (with theory) → INCLUSIVE pass
        passed, t_stat, threshold = chordia_gate(0.30, 100, has_theory=True)
        assert passed is True
        assert t_stat == pytest.approx(3.00, abs=1e-9)
        assert threshold == 3.00

    def test_with_theory_above_boundary_passes(self):
        # sharpe=0.40, N=100 → t=4.0 > 3.00
        passed, t_stat, threshold = chordia_gate(0.40, 100, has_theory=True)
        assert passed is True
        assert t_stat == pytest.approx(4.0, abs=1e-9)
        assert threshold == 3.00

    def test_with_theory_below_boundary_fails(self):
        # sharpe=0.25, N=100 → t=2.5 < 3.00
        passed, t_stat, _threshold = chordia_gate(0.25, 100, has_theory=True)
        assert passed is False
        assert t_stat == pytest.approx(2.5, abs=1e-9)

    def test_without_theory_at_boundary_passes(self):
        # sharpe=0.379, N=100 → t=3.79 == threshold (no theory)
        passed, t_stat, threshold = chordia_gate(0.379, 100, has_theory=False)
        assert passed is True
        assert t_stat == pytest.approx(3.79, abs=1e-9)
        assert threshold == 3.79

    def test_without_theory_just_below_boundary_fails(self):
        # sharpe=0.378, N=100 → t=3.78 < 3.79
        passed, t_stat, threshold = chordia_gate(0.378, 100, has_theory=False)
        assert passed is False

    def test_threshold_changes_verdict(self):
        # A strategy with sharpe=0.32, N=100 → t=3.2:
        # - PASSES with theory (3.2 >= 3.00)
        # - FAILS without theory (3.2 < 3.79)
        passed_with, _t1, _th1 = chordia_gate(0.32, 100, has_theory=True)
        passed_without, _t2, _th2 = chordia_gate(0.32, 100, has_theory=False)
        assert passed_with is True
        assert passed_without is False


class TestSyntheticDistributions:
    """Realistic per-trade-Sharpe scenarios from the project's distribution."""

    def test_high_variance_strategy(self):
        # High-variance: many trades, low per-trade Sharpe but enough N to clear
        # the t-statistic. sharpe=0.15, N=600 → t = 0.15 * sqrt(600) ≈ 3.674
        passed_with, t_stat, _ = chordia_gate(0.15, 600, has_theory=True)
        assert passed_with is True  # 3.674 >= 3.00
        assert t_stat == pytest.approx(0.15 * math.sqrt(600), abs=1e-9)
        # Without theory, the same N is not enough.
        passed_without, _, _ = chordia_gate(0.15, 600, has_theory=False)
        assert passed_without is False  # 3.674 < 3.79

    def test_low_variance_strategy(self):
        # Low-variance: fewer trades but stronger per-trade Sharpe.
        # sharpe=0.45, N=80 → t = 0.45 * sqrt(80) ≈ 4.025
        passed, t_stat, _ = chordia_gate(0.45, 80, has_theory=True)
        assert passed is True  # 4.025 >= 3.00
        assert t_stat == pytest.approx(0.45 * math.sqrt(80), abs=1e-9)

    def test_noise_strategy(self):
        # A noise-floor strategy: sharpe=0.05, N=300 → t ≈ 0.866
        # Should fail BOTH thresholds.
        passed_with, t_stat, _ = chordia_gate(0.05, 300, has_theory=True)
        passed_without, _, _ = chordia_gate(0.05, 300, has_theory=False)
        assert passed_with is False
        assert passed_without is False
        assert t_stat < CHORDIA_T_WITH_THEORY


_FULL_FIELDS_YAML = """\
version: 1
default_has_theory: false
audit_freshness_days: 90
audits:
  - strategy_id: TEST_FULL
    audit_date: 2026-05-03
    audit_reaffirmed_date: 2026-05-04
    verdict: PASS_CHORDIA
    t_stat: 4.361
    t_stat_source: in-memory float64
    t_stat_csv_recompute: 4.323
    oos_n: 49
    oos_power: 0.30
"""

_MINIMAL_FIELDS_YAML = """\
version: 1
default_has_theory: false
audit_freshness_days: 90
audits:
  - strategy_id: TEST_MIN
    audit_date: 2026-05-01
    verdict: PASS_CHORDIA
"""

_OUT_OF_RANGE_POWER_YAML = """\
version: 1
default_has_theory: false
audit_freshness_days: 90
audits:
  - strategy_id: TEST_OOR
    audit_date: 2026-05-01
    verdict: PASS_CHORDIA
    oos_power: 1.5
"""

_REAFFIRM_AS_STRING_YAML = """\
version: 1
default_has_theory: false
audit_freshness_days: 90
audits:
  - strategy_id: TEST_STR
    audit_date: 2026-05-01
    audit_reaffirmed_date: '2026-05-04'
    verdict: PASS_CHORDIA
"""


class TestLoadChordiaAuditEntryAddendum:
    """Schema close for PR #221 evidence-auditor finding.

    YAML carries 6 addendum fields (4 from PR #213 + 1 from PR #221 merge +
    ``t_stat`` peer); loader must populate them on the dataclass instead of
    silently dropping them. Allocator behaviour is unaffected — these tests
    exercise schema, not validation.
    """

    def test_full_fields_round_trip(self, tmp_path):
        path = tmp_path / "chordia_audit_log.yaml"
        path.write_text(_FULL_FIELDS_YAML, encoding="utf-8")
        log = load_chordia_audit_log(path)
        entry = log.entries["TEST_FULL"]
        assert entry.audit_date == date(2026, 5, 3)
        assert entry.audit_reaffirmed_date == date(2026, 5, 4)
        assert entry.verdict == "PASS_CHORDIA"
        assert entry.t_stat == pytest.approx(4.361)
        assert entry.t_stat_source == "in-memory float64"
        assert entry.t_stat_csv_recompute == pytest.approx(4.323)
        assert entry.oos_n == 49
        assert entry.oos_power == pytest.approx(0.30)

    def test_minimal_fields_default_to_none(self, tmp_path):
        # Backward-compat: a row without any addendum fields must still load
        # cleanly and every new field must default to None.
        path = tmp_path / "chordia_audit_log.yaml"
        path.write_text(_MINIMAL_FIELDS_YAML, encoding="utf-8")
        log = load_chordia_audit_log(path)
        entry = log.entries["TEST_MIN"]
        assert entry.audit_date == date(2026, 5, 1)
        assert entry.verdict == "PASS_CHORDIA"
        assert entry.t_stat is None
        assert entry.t_stat_source is None
        assert entry.t_stat_csv_recompute is None
        assert entry.oos_n is None
        assert entry.oos_power is None
        assert entry.audit_reaffirmed_date is None

    def test_oos_power_out_of_range_warns_and_accepts(self, tmp_path, caplog):
        path = tmp_path / "chordia_audit_log.yaml"
        path.write_text(_OUT_OF_RANGE_POWER_YAML, encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="trading_app.chordia"):
            log = load_chordia_audit_log(path)
        entry = log.entries["TEST_OOR"]
        # Stage spec § Decision 2: log.warning + accept; allocator does not
        # consume oos_power, so out-of-range must NOT fail-closed.
        assert entry.oos_power == pytest.approx(1.5)
        assert any("oos_power=1.5" in rec.getMessage() and "TEST_OOR" in rec.getMessage() for rec in caplog.records), (
            f"expected oos_power range-warning; got {[r.getMessage() for r in caplog.records]}"
        )

    def test_audit_reaffirmed_date_string_coerced_to_date(self, tmp_path):
        # YAML quotes ISO dates in single-quoted form, yielding str rather
        # than native date — coercion path must mirror the existing
        # audit_date handling (str -> date.fromisoformat).
        path = tmp_path / "chordia_audit_log.yaml"
        path.write_text(_REAFFIRM_AS_STRING_YAML, encoding="utf-8")
        log = load_chordia_audit_log(path)
        entry = log.entries["TEST_STR"]
        assert entry.audit_reaffirmed_date == date(2026, 5, 4)
        assert isinstance(entry.audit_reaffirmed_date, date)
