"""Tests for scripts.tools.estimate_k_budget and the MinBTL drift check.

Grounding citations (institutional-rigor.md § 7):
- Bailey et al 2013 ``Pseudo-Mathematics`` p.8 — explicit worked example:
  5 years of data, E[max_N]=1.0, the rule is "no more than 45 trials"
  per the tight middle expression of Eq. 6. Our doctrine uses the LOOSE
  upper bound (Eq. 6 RHS, ``2*Ln[N]/E^2``) per Criterion 2; the loose
  form gives a different (stricter) cap. Both are valid Bailey 2013
  outputs — we test against the doctrine-cited loose form.
- ``pre_registered_criteria.md`` Criterion 2 — worked bounds:
    MNQ/MES @ 6.65yr -> N <= 27 strict E=1.0 (loose form)
    MGC   @ 2.70yr -> N <= 3 strict E=1.0 (loose form)
- ``scripts/tools/minbtl_retro_report.strict_bailey_n`` — canonical
  implementation (inverse direction). The tests below cross-check
  ``required_minbtl_years`` against it to prove the inverse round-trips.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.tools.estimate_k_budget import (
    CLEAN_YEARS_BY_INSTRUMENT,
    LOCKED_OPERATIONAL_CAP,
    LOCKED_PROXY_CAP,
    KBudgetReport,
    check_hypothesis_file,
    estimate_k_budget,
    load_hypothesis,
    required_minbtl_years,
)
from scripts.tools.minbtl_retro_report import strict_bailey_n

# ---------------------------------------------------------------------------
# 1. required_minbtl_years — pure math against Bailey worked example
# ---------------------------------------------------------------------------


class TestRequiredMinBTLYears:
    def test_bailey_loose_form_at_n_eq_45_e_eq_1(self) -> None:
        """Bailey 2013 Eq. 6 RHS: 2*Ln[45]/1^2 ~ 7.61 years.

        Doctrine-cited form. NOT the paper's quoted "45 trials at 5yr"
        rule — that uses the tight middle expression. See module docstring.
        """
        result = required_minbtl_years(45, e_max=1.0)
        assert result == pytest.approx(2.0 * math.log(45), abs=1e-9)
        assert 7.6 < result < 7.7

    @pytest.mark.parametrize(
        "n,e_max,expected",
        [
            (10, 1.0, 2.0 * math.log(10)),  # 4.605
            (100, 1.0, 2.0 * math.log(100)),  # 9.210
            (300, 1.0, 2.0 * math.log(300)),  # 11.41 (the locked cap)
            (2000, 1.0, 2.0 * math.log(2000)),  # 15.20 (proxy cap)
            (35000, 1.0, 2.0 * math.log(35000)),  # 20.93 (the bad April 2026 brute force)
            # E[max_N] scaling: higher noise floor -> shorter horizon required
            (45, 2.0, 2.0 * math.log(45) / 4.0),  # /E^2 = /4
            (45, 0.5, 2.0 * math.log(45) / 0.25),  # /E^2 = /0.25
        ],
    )
    def test_formula_matches_2lnN_over_Esq(self, n: int, e_max: float, expected: float) -> None:
        result = required_minbtl_years(n, e_max=e_max)
        assert result == pytest.approx(expected, abs=1e-9)

    def test_n_eq_0_returns_zero(self) -> None:
        """Audit-only N=0 marker → MinBTL=0 (no hypothesis to bound)."""
        assert required_minbtl_years(0) == 0.0

    def test_n_eq_1_returns_zero(self) -> None:
        """Pathway B K=1 → ln(1)=0 → MinBTL=0."""
        assert required_minbtl_years(1) == 0.0
        assert required_minbtl_years(1, e_max=1.5) == 0.0

    def test_negative_n_raises(self) -> None:
        with pytest.raises(ValueError):
            required_minbtl_years(-1)

    def test_nonpositive_e_max_raises(self) -> None:
        with pytest.raises(ValueError):
            required_minbtl_years(10, e_max=0.0)
        with pytest.raises(ValueError):
            required_minbtl_years(10, e_max=-0.5)

    def test_inverse_round_trip_with_strict_bailey_n(self) -> None:
        """``strict_bailey_n`` and ``required_minbtl_years`` must agree.

        strict_bailey_n(horizon, E) returns max N such that MinBTL <= horizon.
        Therefore required_minbtl_years(N_max, E) <= horizon AND
        required_minbtl_years(N_max+1, E) > horizon (for E s.t. N>=2).
        """
        for horizon in (2.7, 5.0, 6.65, 16.0):
            n_max = strict_bailey_n(horizon, e_max=1.0)
            if n_max >= 2:
                assert required_minbtl_years(n_max) <= horizon + 1e-9
            # The next integer up MUST exceed the horizon (by ~1 epsilon)
            # because 2*ln is strictly increasing.
            assert required_minbtl_years(n_max + 1) > horizon - 1e-9


# ---------------------------------------------------------------------------
# 2. estimate_k_budget — full verdict pipeline
# ---------------------------------------------------------------------------


class TestEstimateKBudget:
    def test_pass_on_real_phase0_mnq_prereg(self) -> None:
        """Reproduces the comment in 2026-04-09-mnq-comprehensive.yaml:
        ``# Bailey: N=12, MinBTL=4.97yr < 6.66yr → PASS``."""
        report = estimate_k_budget("MNQ", n_trials=12)
        assert report.passed is True
        assert report.verdict == "PASS"
        assert report.minbtl_years_required == pytest.approx(4.97, abs=0.01)
        assert report.n_max_at_horizon == 27  # Criterion 2 worked bound

    def test_fail_horizon_violation_mgc(self) -> None:
        """MGC has 2.7yr; N=50 needs 7.82yr."""
        report = estimate_k_budget("MGC", n_trials=50)
        assert report.passed is False
        assert "horizon" in report.verdict.lower()
        assert report.n_max_at_horizon == 3
        assert any("reduce N" in n.lower() or "<= 3" in n for n in report.notes)

    def test_fail_operational_cap(self) -> None:
        """N > 300 on clean data triggers cap violation even when horizon ok."""
        # Hypothetical: a 100-yr horizon would make any N viable on horizon
        # grounds; cap is the binding gate.
        report = estimate_k_budget("MNQ", n_trials=LOCKED_OPERATIONAL_CAP + 1)
        assert report.passed is False
        assert "cap" in report.verdict.lower() or "horizon" in report.verdict.lower()

    def test_proxy_cap_widens_limit(self) -> None:
        """proxy_extended=True swaps in the N<=2000 cap."""
        report = estimate_k_budget("MNQ", n_trials=500, proxy_extended=True)
        # 500 is over horizon for MNQ (requires 12.43yr) but under proxy cap.
        # Operational cap not violated; horizon IS violated.
        assert report.operational_cap == LOCKED_PROXY_CAP
        # 500 < 2000 so cap_ok; 12.43 > 6.65 so horizon fails -> overall fail
        assert report.passed is False
        assert report.minbtl_years_required > report.clean_years

    def test_unknown_instrument_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown instrument"):
            estimate_k_budget("XYZ", n_trials=10)

    def test_e_max_relaxation(self) -> None:
        """Higher E[max_N] (looser noise floor) makes more trials viable."""
        n = 100
        report_strict = estimate_k_budget("MNQ", n_trials=n, e_max=1.0)
        report_loose = estimate_k_budget("MNQ", n_trials=n, e_max=1.5)
        assert report_loose.minbtl_years_required < report_strict.minbtl_years_required
        # E=1.5 should allow more trials at any horizon
        assert report_loose.n_max_at_horizon > report_strict.n_max_at_horizon

    def test_report_as_dict_round_trip(self) -> None:
        """KBudgetReport.as_dict() preserves all fields for JSON / MCP."""
        report = estimate_k_budget("MNQ", n_trials=12)
        d = report.as_dict()
        assert d["instrument"] == "MNQ"
        assert d["n_trials"] == 12
        assert d["passed"] is True
        assert isinstance(d["notes"], list)


# ---------------------------------------------------------------------------
# 3. load_hypothesis + check_hypothesis_file — YAML parsing
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestLoadHypothesis:
    def test_legacy_total_expected_trials_top_level(self, tmp_path: Path) -> None:
        yaml = "metadata:\n  name: test\ntotal_expected_trials: 12\nhypotheses:\n  - id: 1\n    scope: {instruments: [MNQ]}\n"
        f = tmp_path / "2026-05-12-test.yaml"
        _write_yaml(f, yaml)
        summary = load_hypothesis(f)
        assert summary.n_trials == 12
        assert summary.instruments == ("MNQ",)

    def test_primary_selection_trials_alternate_key(self, tmp_path: Path) -> None:
        yaml = (
            "trial_budget:\n  primary_selection_trials: 1296\nhypotheses:\n  - scope: {instruments: [MNQ, MES, MGC]}\n"
        )
        f = tmp_path / "2026-05-12-vwap-scan.yaml"
        _write_yaml(f, yaml)
        summary = load_hypothesis(f)
        assert summary.n_trials == 1296
        assert summary.instruments == ("MES", "MGC", "MNQ")

    def test_n_trials_pathway_b(self, tmp_path: Path) -> None:
        yaml = "statistical_plan:\n  n_trials: 1\nhypotheses:\n  - scope: {instruments: [MGC]}\n"
        f = tmp_path / "2026-05-12-individual.yaml"
        _write_yaml(f, yaml)
        summary = load_hypothesis(f)
        assert summary.n_trials == 1

    def test_missing_trial_count_returns_none(self, tmp_path: Path) -> None:
        yaml = "metadata:\n  name: stub\nhypotheses:\n  - scope: {instruments: [MNQ]}\n"
        f = tmp_path / "2026-05-12-stub.yaml"
        _write_yaml(f, yaml)
        summary = load_hypothesis(f)
        assert summary.n_trials is None
        assert summary.instruments == ("MNQ",)

    def test_total_expected_trials_wins_over_alternates(self, tmp_path: Path) -> None:
        """Priority order: total_expected_trials > primary > n_trials."""
        yaml = "total_expected_trials: 12\nprimary_selection_trials: 1296\nn_trials: 1\n"
        f = tmp_path / "2026-05-12-multi.yaml"
        _write_yaml(f, yaml)
        summary = load_hypothesis(f)
        assert summary.n_trials == 12

    def test_proxy_filename_heuristic(self, tmp_path: Path) -> None:
        yaml = "total_expected_trials: 100\nhypotheses:\n  - scope: {instruments: [MGC]}\n"
        f = tmp_path / "2026-05-12-gc-proxy-something.yaml"
        _write_yaml(f, yaml)
        summary = load_hypothesis(f)
        assert summary.proxy_extended is True

    def test_check_hypothesis_file_skips_unknown_instruments(self, tmp_path: Path) -> None:
        yaml = "total_expected_trials: 10\nhypotheses:\n  - scope: {instruments: [6A, 6B]}\n"
        f = tmp_path / "2026-05-12-fx.yaml"
        _write_yaml(f, yaml)
        # 6A/6B not in CLEAN_YEARS_BY_INSTRUMENT — no reports generated
        reports = check_hypothesis_file(f)
        assert reports == []

    def test_check_hypothesis_file_returns_one_per_instrument(self, tmp_path: Path) -> None:
        yaml = "total_expected_trials: 10\nhypotheses:\n  - scope: {instruments: [MNQ, MES, MGC]}\n"
        f = tmp_path / "2026-05-12-multi.yaml"
        _write_yaml(f, yaml)
        reports = check_hypothesis_file(f)
        # N=10 fits MNQ/MES (6.65yr horizon, requires 4.61yr) but fails MGC
        # (2.70yr horizon, requires 4.61yr).
        assert len(reports) == 3
        by_inst = {r.instrument: r for r in reports}
        assert by_inst["MNQ"].passed is True
        assert by_inst["MES"].passed is True
        assert by_inst["MGC"].passed is False


# ---------------------------------------------------------------------------
# 4. Drift check — sentinel-date enforcement
# ---------------------------------------------------------------------------


class TestDriftCheck:
    """Sentinel: 2026-05-12 (matches the drift-check definition).

    Pre-sentinel files emit advisories (return [] / pass). Post-sentinel
    files contribute to violations on operational-cap, horizon, or
    missing-N. This mirrors the pooled-finding-rule rollout pattern.
    """

    def _setup(self, tmp_path: Path):
        from pipeline import check_drift

        hyp_dir = tmp_path / "docs" / "audit" / "hypotheses"
        hyp_dir.mkdir(parents=True)
        return check_drift, hyp_dir

    def test_pre_sentinel_violation_is_advisory_only(self, tmp_path: Path, capsys) -> None:
        check_drift, hyp_dir = self._setup(tmp_path)
        # Pre-sentinel filename, deliberately under-budgeted
        yaml = "total_expected_trials: 50\nhypotheses:\n  - scope: {instruments: [MGC]}\n"
        _write_yaml(hyp_dir / "2026-04-09-old-under-budget.yaml", yaml)
        with patch.object(check_drift, "PROJECT_ROOT", tmp_path):
            violations = check_drift.check_hypothesis_minbtl_compliance()
        assert violations == [], f"Pre-sentinel file should be advisory only, got: {violations}"
        # Advisory must be printed
        captured = capsys.readouterr()
        assert "advisory" in captured.out.lower()

    def test_post_sentinel_horizon_violation_blocks(self, tmp_path: Path) -> None:
        check_drift, hyp_dir = self._setup(tmp_path)
        yaml = "total_expected_trials: 50\nhypotheses:\n  - scope: {instruments: [MGC]}\n"
        _write_yaml(hyp_dir / "2026-06-01-new-under-budget.yaml", yaml)
        with patch.object(check_drift, "PROJECT_ROOT", tmp_path):
            violations = check_drift.check_hypothesis_minbtl_compliance()
        assert len(violations) == 1
        assert "MGC" in violations[0]
        assert "requires" in violations[0]

    def test_post_sentinel_operational_cap_blocks(self, tmp_path: Path) -> None:
        check_drift, hyp_dir = self._setup(tmp_path)
        # 301 exceeds clean N<=300 cap regardless of horizon
        yaml = f"total_expected_trials: {LOCKED_OPERATIONAL_CAP + 1}\nhypotheses:\n  - scope: {{instruments: [MNQ]}}\n"
        _write_yaml(hyp_dir / "2026-06-01-too-many.yaml", yaml)
        with patch.object(check_drift, "PROJECT_ROOT", tmp_path):
            violations = check_drift.check_hypothesis_minbtl_compliance()
        assert len(violations) == 1
        assert "operational cap" in violations[0].lower()

    def test_post_sentinel_missing_n_blocks(self, tmp_path: Path) -> None:
        check_drift, hyp_dir = self._setup(tmp_path)
        yaml = "metadata:\n  name: stub\nhypotheses:\n  - scope: {instruments: [MNQ]}\n"
        _write_yaml(hyp_dir / "2026-06-01-no-n.yaml", yaml)
        with patch.object(check_drift, "PROJECT_ROOT", tmp_path):
            violations = check_drift.check_hypothesis_minbtl_compliance()
        assert len(violations) == 1
        assert "Criterion 1" in violations[0]

    def test_post_sentinel_pass_within_bound(self, tmp_path: Path) -> None:
        check_drift, hyp_dir = self._setup(tmp_path)
        # MNQ N=12, MinBTL=4.97yr < 6.65yr clean — clean pass
        yaml = "total_expected_trials: 12\nhypotheses:\n  - scope: {instruments: [MNQ]}\n"
        _write_yaml(hyp_dir / "2026-06-01-good.yaml", yaml)
        with patch.object(check_drift, "PROJECT_ROOT", tmp_path):
            violations = check_drift.check_hypothesis_minbtl_compliance()
        assert violations == []

    def test_pathway_b_k_eq_1_passes_post_sentinel(self, tmp_path: Path) -> None:
        check_drift, hyp_dir = self._setup(tmp_path)
        # K=1 → MinBTL=0 trivially passes even on MGC's short horizon
        yaml = "total_expected_trials: 1\nhypotheses:\n  - scope: {instruments: [MGC]}\n"
        _write_yaml(hyp_dir / "2026-06-01-pathway-b.yaml", yaml)
        with patch.object(check_drift, "PROJECT_ROOT", tmp_path):
            violations = check_drift.check_hypothesis_minbtl_compliance()
        assert violations == []

    def test_audit_only_n_eq_0_passes_post_sentinel(self, tmp_path: Path) -> None:
        check_drift, hyp_dir = self._setup(tmp_path)
        yaml = "n_trials: 0\nhypotheses:\n  - scope: {instruments: [MGC]}\n"
        _write_yaml(hyp_dir / "2026-06-01-audit-only.yaml", yaml)
        with patch.object(check_drift, "PROJECT_ROOT", tmp_path):
            violations = check_drift.check_hypothesis_minbtl_compliance()
        assert violations == []

    def test_unknown_instrument_always_advisory(self, tmp_path: Path, capsys) -> None:
        check_drift, hyp_dir = self._setup(tmp_path)
        yaml = "total_expected_trials: 100\nhypotheses:\n  - scope: {instruments: [6A]}\n"
        # POST-sentinel filename — even so, unknown instrument is advisory
        _write_yaml(hyp_dir / "2026-06-01-fx.yaml", yaml)
        with patch.object(check_drift, "PROJECT_ROOT", tmp_path):
            violations = check_drift.check_hypothesis_minbtl_compliance()
        assert violations == []


# ---------------------------------------------------------------------------
# 5. MCP-layer wrapper
# ---------------------------------------------------------------------------


class TestMCPTool:
    def test_inline_pass(self) -> None:
        from scripts.tools.research_catalog_mcp_server import _estimate_k_budget_tool

        out = _estimate_k_budget_tool(instrument="MNQ", n_trials=12)
        assert out["mode"] == "inline"
        assert out["passed"] is True
        assert out["report"]["verdict"] == "PASS"

    def test_inline_fail(self) -> None:
        from scripts.tools.research_catalog_mcp_server import _estimate_k_budget_tool

        out = _estimate_k_budget_tool(instrument="MGC", n_trials=50)
        assert out["passed"] is False
        assert "horizon" in out["report"]["verdict"].lower()

    def test_inline_proxy_extended(self) -> None:
        from scripts.tools.research_catalog_mcp_server import _estimate_k_budget_tool

        out = _estimate_k_budget_tool(instrument="MNQ", n_trials=500, proxy_extended=True)
        assert out["report"]["operational_cap"] == LOCKED_PROXY_CAP

    def test_invalid_call_shape_raises(self) -> None:
        from scripts.tools.research_catalog_mcp_server import _estimate_k_budget_tool

        with pytest.raises(ValueError, match="either"):
            _estimate_k_budget_tool(instrument="MNQ", n_trials=12, hypothesis_id="x")
        with pytest.raises(ValueError):
            _estimate_k_budget_tool(instrument="MNQ")  # missing n_trials
        with pytest.raises(ValueError, match="Unknown instrument"):
            _estimate_k_budget_tool(instrument="XYZ", n_trials=10)

    def test_authority_field_present(self) -> None:
        """MCP callers see the doctrine citation — institutional-rigor §7."""
        from scripts.tools.research_catalog_mcp_server import _estimate_k_budget_tool

        out = _estimate_k_budget_tool(instrument="MNQ", n_trials=12)
        assert "Criterion 2" in out["authority"]
        assert "bailey" in out["authority"].lower()
