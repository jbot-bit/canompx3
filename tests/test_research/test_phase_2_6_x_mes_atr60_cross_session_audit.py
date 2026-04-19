"""Tests for research/phase_2_6_x_mes_atr60_cross_session_audit.py.

Covers:
- Canonical delegation (all thresholds sourced from canonical modules)
- subset_t math
- bh_fdr correctness (monotonic + edge cases)
- K=6 cell list matches pre-reg yaml exactly
- SESSION_CATALOG whitelist enforcement
- End-to-end smoke test
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research import phase_2_6_x_mes_atr60_cross_session_audit as mod


class TestCanonicalDelegation:
    def test_imports_c4_from_mode_a_revalidation(self):
        from research.mode_a_revalidation_active_setups import C4_T_WITH_THEORY
        assert mod.C4_T_WITH_THEORY is C4_T_WITH_THEORY

    def test_imports_c7_c9_from_mode_a_revalidation(self):
        from research.mode_a_revalidation_active_setups import (
            C7_MIN_N, C9_ERA_THRESHOLD, C9_MIN_N_PER_ERA,
        )
        assert mod.C7_MIN_N is C7_MIN_N
        assert mod.C9_ERA_THRESHOLD is C9_ERA_THRESHOLD
        assert mod.C9_MIN_N_PER_ERA is C9_MIN_N_PER_ERA

    def test_imports_holdout_from_trading_app(self):
        from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
        assert mod.HOLDOUT_SACRED_FROM is HOLDOUT_SACRED_FROM

    def test_c6_threshold_annotated(self):
        src = (PROJECT_ROOT / "research" / "phase_2_6_x_mes_atr60_cross_session_audit.py").read_text()
        assert "@research-source" in src
        assert "Criterion 6" in src
        assert mod.C6_WFE_THRESHOLD == 0.50

    def test_no_inlined_holdout_date(self):
        src = (PROJECT_ROOT / "research" / "phase_2_6_x_mes_atr60_cross_session_audit.py").read_text()
        assert "'2026-01-01'" not in src
        assert "date(2026, 1, 1)" not in src


class TestCellListMatchesPreReg:
    """Pre-reg yaml K=6 cells must match CELLS in script exactly — drift guard."""

    @staticmethod
    def _pre_reg():
        path = PROJECT_ROOT / "docs" / "audit" / "hypotheses" / "2026-04-19-x-mes-atr60-cross-session-extension-v1.yaml"
        with path.open() as f:
            return yaml.safe_load(f)

    def test_k_matches(self):
        pr = self._pre_reg()
        assert pr["total_hypothesis_count"] == 6
        assert len(mod.CELLS) == 6

    def test_cells_map_to_hypotheses(self):
        pr = self._pre_reg()
        pr_cells = [
            (h["scope"]["sessions"][0], h["scope"]["rr_targets"][0])
            for h in pr["hypotheses"]
        ]
        script_cells = [(c["session"], c["rr"]) for c in mod.CELLS]
        assert pr_cells == script_cells, (
            f"Script cells drifted from pre-reg.\n"
            f"Pre-reg: {pr_cells}\nScript: {script_cells}"
        )


class TestSubsetT:
    def test_positive_case(self):
        # ExpR=0.2, sd=1.0, n=100 → t = 0.2 * 10 = 2.0
        assert math.isclose(mod.subset_t(0.2, 1.0, 100), 2.0, abs_tol=1e-9)

    def test_none_returns_nan(self):
        assert math.isnan(mod.subset_t(None, 1.0, 100))
        assert math.isnan(mod.subset_t(0.1, None, 100))

    def test_zero_sd_returns_nan(self):
        assert math.isnan(mod.subset_t(0.1, 0.0, 100))


class TestBhFdr:
    def test_all_pass_when_all_sig(self):
        """All p=0.001 at K=3, q=0.05 → all pass."""
        passes = mod.bh_fdr([0.001, 0.001, 0.001], q=0.05)
        assert passes == [True, True, True]

    def test_none_pass_when_all_large(self):
        passes = mod.bh_fdr([0.5, 0.5, 0.5], q=0.05)
        assert passes == [False, False, False]

    def test_step_down_order(self):
        """BH ranks by p; largest passing rank is the cut.
        p=[0.001, 0.02, 0.3] at q=0.05: rank-1 p=0.001 <= 1/3*0.05=0.0167 PASS,
        rank-2 p=0.02 <= 2/3*0.05=0.033 PASS, rank-3 p=0.3 > 0.05 FAIL.
        """
        passes = mod.bh_fdr([0.001, 0.02, 0.3], q=0.05)
        assert passes == [True, True, False]

    def test_nan_handled_as_fail(self):
        passes = mod.bh_fdr([0.001, float("nan")], q=0.05)
        assert passes[0] is True
        assert passes[1] is False


class TestSessionWhitelist:
    def test_invalid_session_raises(self):
        bad_cell = {"id": 99, "session": "NOT_A_REAL_SESSION", "rr": 1.0}
        with pytest.raises(ValueError, match="SESSION_CATALOG"):
            mod._spec(bad_cell)

    def test_all_pre_reg_sessions_are_canonical(self):
        from pipeline.dst import SESSION_CATALOG
        for c in mod.CELLS:
            assert c["session"] in SESSION_CATALOG


@pytest.mark.integration
class TestEndToEnd:
    def test_main_runs_all_6_cells(self, tmp_path, monkeypatch, capsys):
        from pipeline.paths import GOLD_DB_PATH
        if not Path(GOLD_DB_PATH).exists():
            pytest.skip("gold.db not available")
        monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path)
        rc = mod.main()
        assert rc == 0
        csv = tmp_path / "phase_2_6_x_mes_atr60_cross_session_audit.csv"
        assert csv.exists()
        df = pd.read_csv(csv)
        assert len(df) == 6
        # All cells should have results (no error rows)
        assert "error" not in df.columns or df["error"].isna().all()
        # Every cell has a verdict
        assert df["verdict"].notna().all()
        # Captured stdout has expected headers
        captured = capsys.readouterr()
        assert "PHASE 2.6" in captured.out
        assert "BH-FDR K=6" in captured.out
