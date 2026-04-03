"""Tests for governance tools: trace, research claim validator, stale-doc scanner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.trace import (
    Finding,
    GovernanceDecision,
    TraceReport,
    get_git_state,
)
from scripts.tools.research_claim_validator import (
    ValidationResult,
    classify_sample_size,
    validate_claim,
)
from scripts.tools.stale_doc_scanner import (
    ScanResult,
    check_as_of_dates,
    check_e0_references,
    check_hardcoded_counts,
    check_instrument_status,
    check_strategy_counts,
)


# -----------------------------------------------------------------------
# pipeline/trace.py tests
# -----------------------------------------------------------------------


class TestGovernanceDecision:
    def test_enum_values(self):
        assert GovernanceDecision.VALID.value == "VALID"
        assert GovernanceDecision.INVALID.value == "INVALID"
        assert GovernanceDecision.REGIME_ONLY.value == "REGIME_ONLY"
        assert GovernanceDecision.STALE.value == "STALE"
        assert GovernanceDecision.UNSUPPORTED.value == "UNSUPPORTED"
        assert GovernanceDecision.BLOCKED.value == "BLOCKED"

    def test_string_comparison(self):
        assert GovernanceDecision.VALID == "VALID"


class TestTraceReport:
    def test_defaults(self):
        tr = TraceReport(task="test task")
        assert tr.task == "test task"
        assert len(tr.trace_id) == 12
        assert tr.governance_decision == "VALID"
        assert tr.findings == []

    def test_add_finding(self):
        tr = TraceReport(task="test")
        tr.add_finding("check1", "PASS", "all good")
        tr.add_finding("check2", "FAIL", "bad", data={"count": 0})
        assert len(tr.findings) == 2
        assert tr.findings[0].status == "PASS"
        assert tr.findings[1].data == {"count": 0}

    def test_has_failures(self):
        tr = TraceReport(task="test")
        assert not tr.has_failures()
        tr.add_finding("check1", "PASS", "ok")
        assert not tr.has_failures()
        tr.add_finding("check2", "FAIL", "bad")
        assert tr.has_failures()

    def test_has_warnings(self):
        tr = TraceReport(task="test")
        assert not tr.has_warnings()
        tr.add_finding("check1", "WARN", "hmm")
        assert tr.has_warnings()

    def test_summary(self):
        tr = TraceReport(task="test")
        tr.add_finding("a", "PASS", "ok")
        tr.add_finding("b", "PASS", "ok")
        tr.add_finding("c", "FAIL", "bad")
        tr.add_finding("d", "WARN", "hmm")
        assert tr.summary() == "2 PASS, 1 FAIL, 1 WARN — VALID"

    def test_to_dict(self):
        tr = TraceReport(task="test")
        tr.add_finding("check1", "PASS", "ok")
        d = tr.to_dict()
        assert d["task"] == "test"
        assert len(d["findings"]) == 1
        assert d["findings"][0]["check"] == "check1"

    def test_write_roundtrip(self, tmp_path: Path):
        tr = TraceReport(task="roundtrip test")
        tr.add_finding("check1", "PASS", "ok")
        tr.governance_decision = GovernanceDecision.VALID.value

        path = tr.write(directory=tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

        data = json.loads(path.read_text())
        assert data["task"] == "roundtrip test"
        assert len(data["findings"]) == 1
        assert data["governance_decision"] == "VALID"

    def test_write_sanitizes_filename(self, tmp_path: Path):
        tr = TraceReport(task="bad/chars\\here & more!")
        path = tr.write(directory=tmp_path)
        assert path.exists()
        # Filename should not contain unsafe characters
        assert "/" not in path.name
        assert "\\" not in path.name


class TestGetGitState:
    def test_returns_dict(self):
        state = get_git_state()
        assert isinstance(state, dict)
        # Should have branch and commit (we're in a git repo)
        assert "branch" in state or "error" in state


# -----------------------------------------------------------------------
# scripts/tools/research_claim_validator.py tests
# -----------------------------------------------------------------------


class TestClassifySampleSize:
    def test_invalid(self):
        assert classify_sample_size(0) == "INVALID"
        assert classify_sample_size(29) == "INVALID"

    def test_regime(self):
        assert classify_sample_size(30) == "REGIME"
        assert classify_sample_size(99) == "REGIME"

    def test_preliminary(self):
        assert classify_sample_size(100) == "PRELIMINARY"
        assert classify_sample_size(199) == "PRELIMINARY"

    def test_core(self):
        assert classify_sample_size(200) == "CORE"
        assert classify_sample_size(499) == "CORE"

    def test_high_confidence(self):
        assert classify_sample_size(500) == "HIGH-CONFIDENCE"
        assert classify_sample_size(10000) == "HIGH-CONFIDENCE"


class TestValidateClaim:
    def test_valid_claim(self):
        result = validate_claim(
            n=250,
            p_value=0.003,
            mechanism="cost-gated friction filter",
            wfe=0.62,
        )
        assert result.decision == GovernanceDecision.VALID
        assert result.sample_class == "CORE"
        assert not result.reasons  # No blocking reasons

    def test_invalid_sample_size(self):
        result = validate_claim(n=25, p_value=0.001, mechanism="test")
        assert result.decision == GovernanceDecision.INVALID
        assert result.sample_class == "INVALID"

    def test_regime_only(self):
        result = validate_claim(n=50, p_value=0.003, mechanism="test")
        assert result.decision == GovernanceDecision.REGIME_ONLY
        assert result.sample_class == "REGIME"

    def test_invalid_p_value(self):
        result = validate_claim(n=200, p_value=0.06, mechanism="test")
        assert result.decision == GovernanceDecision.INVALID
        assert any("not significant" in r for r in result.reasons)

    def test_low_wfe(self):
        result = validate_claim(n=200, p_value=0.003, mechanism="test", wfe=0.35)
        assert result.decision == GovernanceDecision.UNSUPPORTED
        assert any("WFE" in r for r in result.reasons)

    def test_no_mechanism(self):
        result = validate_claim(n=200, p_value=0.003)
        assert result.decision == GovernanceDecision.UNSUPPORTED
        assert any("mechanism" in r.lower() for r in result.reasons)

    def test_regime_with_low_wfe_becomes_unsupported(self):
        """REGIME_ONLY + low WFE should escalate to UNSUPPORTED."""
        result = validate_claim(n=50, p_value=0.003, mechanism="test", wfe=0.30)
        assert result.decision == GovernanceDecision.UNSUPPORTED

    def test_regime_with_no_mechanism_becomes_unsupported(self):
        """REGIME_ONLY + no mechanism should escalate to UNSUPPORTED."""
        result = validate_claim(n=50, p_value=0.003)
        assert result.decision == GovernanceDecision.UNSUPPORTED

    def test_sensitivity_failed(self):
        result = validate_claim(n=200, p_value=0.003, mechanism="test", sensitivity_passed=False)
        assert result.decision == GovernanceDecision.INVALID
        assert any("sensitivity" in r.lower() for r in result.reasons)

    def test_sensitivity_passed(self):
        result = validate_claim(n=200, p_value=0.003, mechanism="test", sensitivity_passed=True)
        assert result.decision == GovernanceDecision.VALID

    def test_short_time_span_warns(self):
        result = validate_claim(n=200, p_value=0.003, mechanism="test", time_span_years=2.0)
        assert any("regime diversity" in w for w in result.warnings)

    def test_boundary_n_30(self):
        """N=30 is the boundary — should be REGIME, not INVALID."""
        result = validate_claim(n=30, p_value=0.003, mechanism="test")
        assert result.decision == GovernanceDecision.REGIME_ONLY

    def test_boundary_p_005(self):
        """p=0.05 is the boundary — should be INVALID (>= threshold)."""
        result = validate_claim(n=200, p_value=0.05, mechanism="test")
        assert result.decision == GovernanceDecision.INVALID

    def test_boundary_p_just_under(self):
        """p=0.049 should be significant (just barely)."""
        result = validate_claim(n=200, p_value=0.049, mechanism="test")
        assert result.decision == GovernanceDecision.VALID

    def test_multiple_failures_worst_wins(self):
        """When both N and p fail, INVALID wins over REGIME_ONLY."""
        result = validate_claim(n=10, p_value=0.06)
        assert result.decision == GovernanceDecision.INVALID

    def test_bh_k_warning(self):
        result = validate_claim(n=200, p_value=0.003, mechanism="test", bh_k=50)
        assert any("BH FDR" in w for w in result.warnings)

    def test_result_to_dict(self):
        result = validate_claim(n=200, p_value=0.003, mechanism="test")
        d = result.to_dict()
        assert "decision" in d
        assert "sample_class" in d
        assert "reasons" in d
        assert "warnings" in d


# -----------------------------------------------------------------------
# scripts/tools/stale_doc_scanner.py tests
# -----------------------------------------------------------------------


class TestCheckAsOfDates:
    def test_recent_date_no_flag(self):
        result = ScanResult()
        # Use today's date — should not be flagged
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        check_as_of_dates(f"As of {today}", 1, "test.md", result)
        assert len(result.claims) == 0

    def test_old_date_flagged(self):
        result = ScanResult()
        check_as_of_dates("As of 2025-01-01", 1, "test.md", result)
        assert len(result.claims) == 1
        assert result.claims[0].severity == "WARN"

    def test_case_insensitive(self):
        result = ScanResult()
        check_as_of_dates("as of 2025-01-01", 1, "test.md", result)
        assert len(result.claims) == 1


class TestCheckInstrumentStatus:
    def test_dead_instrument_as_active(self):
        result = ScanResult()
        dead = {"MCL", "SIL", "M6E", "MBT", "M2K"}
        check_instrument_status("Active instruments: MGC, MNQ, M2K", 1, "test.md", result, dead)
        assert len(result.claims) == 1
        assert "M2K" in result.claims[0].claim

    def test_live_instruments_clean(self):
        result = ScanResult()
        dead = {"MCL", "SIL", "M6E", "MBT", "M2K"}
        check_instrument_status("Active instruments: MGC, MNQ, MES", 1, "test.md", result, dead)
        assert len(result.claims) == 0

    def test_dead_instrument_in_prose(self):
        """Prose pattern: 'MCL is an active instrument' should be caught."""
        result = ScanResult()
        dead = {"MCL", "SIL", "M6E", "MBT", "M2K"}
        check_instrument_status("MCL is still an active futures contract", 1, "test.md", result, dead)
        assert len(result.claims) == 1

    def test_dead_instrument_marked_dead_not_flagged(self):
        """Lines saying instrument is dead should not be flagged."""
        result = ScanResult()
        dead = {"MCL", "SIL", "M6E", "MBT", "M2K"}
        check_instrument_status("M2K is dead for ORB trading", 1, "test.md", result, dead)
        assert len(result.claims) == 0


class TestCheckE0References:
    def test_e0_active_flagged(self):
        result = ScanResult()
        check_e0_references("E0 is the active entry model to deploy", 1, "test.md", result)
        assert len(result.claims) == 1
        assert result.claims[0].severity == "STALE"

    def test_e0_purged_not_flagged(self):
        result = ScanResult()
        check_e0_references("E0 was purged in Feb 2026", 1, "test.md", result)
        assert len(result.claims) == 0

    def test_e0_in_dead_context(self):
        result = ScanResult()
        check_e0_references("E0 is dead and removed from pipeline", 1, "test.md", result)
        assert len(result.claims) == 0


class TestCheckHardcodedCounts:
    def test_large_count_flagged(self):
        result = ScanResult()
        check_hardcoded_counts("all 78 drift checks passed", 1, "test.md", result)
        assert len(result.claims) == 1

    def test_small_count_ignored(self):
        result = ScanResult()
        check_hardcoded_counts("run 3 checks before committing", 1, "test.md", result)
        assert len(result.claims) == 0


class TestCheckStrategyCounts:
    def test_count_flagged(self):
        result = ScanResult()
        check_strategy_counts("We have 747 validated strategies", 1, "test.md", result)
        assert len(result.claims) == 1
        assert result.claims[0].severity == "WARN"

    def test_small_numbers_ignored(self):
        result = ScanResult()
        check_strategy_counts("We have 3 instruments", 1, "test.md", result)
        assert len(result.claims) == 0


class TestScanResult:
    def test_severity_counts(self):
        r = ScanResult()
        r.add("f.md", 1, "claim1", "actual1", "STALE")
        r.add("f.md", 2, "claim2", "actual2", "WARN")
        r.add("f.md", 3, "claim3", "actual3", "STALE")
        assert r.stale_count == 2
        assert r.warn_count == 1

    def test_summary(self):
        r = ScanResult(files_scanned=10)
        r.add("f.md", 1, "claim1", "actual1", "STALE")
        assert "1 STALE" in r.summary()
        assert "10 files" in r.summary()
