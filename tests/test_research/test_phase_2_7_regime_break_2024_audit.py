"""Tests for research/phase_2_7_regime_break_2024_audit.py.

Covers:
- Canonical delegation (no inlined constants)
- subset_t math edge cases
- classify_2024_flag exhaustive branches (5 flag outcomes)
- Session whitelist enforcement
- End-to-end smoke test
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research import phase_2_7_regime_break_2024_audit as mod


class TestCanonicalDelegation:
    def test_imports_compute_mode_a(self):
        from research.mode_a_revalidation_active_setups import compute_mode_a
        assert mod.compute_mode_a is compute_mode_a

    def test_imports_holdout_policy(self):
        from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
        assert mod.HOLDOUT_SACRED_FROM is HOLDOUT_SACRED_FROM

    def test_valid_sessions_from_catalog(self):
        from pipeline.dst import SESSION_CATALOG
        assert mod._VALID_SESSIONS == frozenset(SESSION_CATALOG.keys())

    def test_no_inlined_dates_other_than_2024_window(self):
        src = (PROJECT_ROOT / "research" / "phase_2_7_regime_break_2024_audit.py").read_text()
        assert "'2026-01-01'" not in src
        # 2024 window is allowed inlined — it's the semantic boundary under test
        assert "date(2024, 1, 1)" in src
        assert "date(2025, 1, 1)" in src

    def test_literature_grounding_cited(self):
        src = (PROJECT_ROOT / "research" / "phase_2_7_regime_break_2024_audit.py").read_text()
        assert "chan_2008_ch7_regime_switching" in src
        assert "Chan 2008" in src


class TestSubsetT:
    def test_standard(self):
        # ExpR=0.1, sd=1.0, n=100 → t=1.0
        assert math.isclose(mod.subset_t(0.1, 1.0, 100), 1.0, abs_tol=1e-9)

    def test_none_returns_nan(self):
        assert math.isnan(mod.subset_t(None, 1.0, 100))
        assert math.isnan(mod.subset_t(0.1, None, 100))

    def test_zero_sd_returns_nan(self):
        assert math.isnan(mod.subset_t(0.1, 0.0, 100))

    def test_n_below_2_returns_nan(self):
        assert math.isnan(mod.subset_t(0.1, 1.0, 1))


class TestClassify2024Flag:
    """Exhaustive flag-branch coverage per classify_2024_flag logic."""

    def test_pure_drag(self):
        # 2024 expr -0.10 (below -0.05 floor), ex2024 lift +0.05 (above 0.03)
        flag = mod.classify_2024_flag(
            expr_full=0.05, expr_ex2024=0.10, expr_2024=-0.10, n_2024=50,
        )
        assert flag == "2024_PURE_DRAG"

    def test_critical(self):
        # ex2024 drop 0.05 (below -0.03)
        flag = mod.classify_2024_flag(
            expr_full=0.20, expr_ex2024=0.15, expr_2024=0.35, n_2024=50,
        )
        assert flag == "2024_CRITICAL"

    def test_neutral(self):
        flag = mod.classify_2024_flag(
            expr_full=0.10, expr_ex2024=0.11, expr_2024=0.08, n_2024=50,
        )
        assert flag == "2024_NEUTRAL"

    def test_mixed_positive_below_threshold(self):
        # delta between 0.03 (exclusive-neutral) and 0.03 (pure-drag cut)
        # requires lift > 0.03 but NOT pure-drag (which requires 2024 <= -0.05)
        flag = mod.classify_2024_flag(
            expr_full=0.08, expr_ex2024=0.12, expr_2024=0.02, n_2024=50,
        )
        # delta=0.04>0.03, 2024=0.02 > -0.05 so not drag, not critical (positive lift)
        # falls to MIXED
        assert flag == "2024_MIXED"

    def test_unevaluable_thin_2024(self):
        flag = mod.classify_2024_flag(
            expr_full=0.10, expr_ex2024=0.12, expr_2024=0.05, n_2024=20,
        )
        assert flag == "2024_UNEVALUABLE"

    def test_unevaluable_missing_full(self):
        flag = mod.classify_2024_flag(
            expr_full=None, expr_ex2024=0.1, expr_2024=0.1, n_2024=50,
        )
        assert flag == "2024_UNEVALUABLE"

    def test_pure_drag_requires_both_conditions(self):
        # 2024 negative BUT ex2024 lift below threshold → not PURE_DRAG
        flag = mod.classify_2024_flag(
            expr_full=0.05, expr_ex2024=0.07, expr_2024=-0.10, n_2024=50,
        )
        assert flag != "2024_PURE_DRAG"  # delta=0.02 < threshold
        assert flag == "2024_NEUTRAL"


class TestSessionWhitelist:
    def test_invalid_session_raises_in_window_stats(self):
        import duckdb
        con = duckdb.connect(":memory:")
        bad_spec = {
            "instrument": "MNQ",
            "orb_label": "NOT_A_SESSION",
            "orb_minutes": 5,
            "entry_model": "E2",
            "confirm_bars": 1,
            "rr_target": 1.5,
            "execution_spec": '{"direction":"long"}',
        }
        with pytest.raises(ValueError, match="SESSION_CATALOG"):
            mod._compute_window_stats(
                con, bad_spec,
                window_sql_clause="o.trading_day < ?",
                window_params=[mod.HOLDOUT_SACRED_FROM],
            )
        con.close()


class TestThresholds:
    def test_drag_threshold_matches_c9_criterion(self):
        """FLAG_DRAG_YR_EXPR = -0.05 must match pre_registered_criteria C9."""
        from research.mode_a_revalidation_active_setups import C9_ERA_THRESHOLD
        assert mod.FLAG_DRAG_YR_EXPR == C9_ERA_THRESHOLD

    def test_unevaluable_matches_rule_3_2(self):
        """Rule 3.2 says N<30 is directional-only."""
        assert mod.FLAG_UNEVALUABLE_MIN_N == 30

    def test_2024_window_boundaries(self):
        from datetime import date
        assert mod.YEAR_2024_START == date(2024, 1, 1)
        assert mod.YEAR_2024_END == date(2025, 1, 1)


@pytest.mark.integration
class TestEndToEnd:
    def test_main_runs_38_lanes(self, tmp_path, monkeypatch, capsys):
        if not Path(mod.GOLD_DB_PATH).exists():
            pytest.skip("gold.db not available")
        monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path)
        rc = mod.main()
        assert rc == 0
        csv = tmp_path / "phase_2_7_regime_break_2024_audit.csv"
        assert csv.exists()
        df = pd.read_csv(csv)
        assert 10 <= len(df) <= 200  # allow active-set growth
        required_cols = {
            "strategy_id", "full_expr", "ex2024_expr", "y2024_expr",
            "delta_expr_ex2024_minus_full", "flag_2024",
            "atr20_pct_2024", "atr20_pct_rest",
        }
        assert required_cols.issubset(df.columns)
        # Flag should be one of known values
        valid_flags = {"2024_NEUTRAL", "2024_PURE_DRAG", "2024_CRITICAL",
                       "2024_MIXED", "2024_UNEVALUABLE", "ERROR"}
        assert set(df["flag_2024"].unique()).issubset(valid_flags)
        captured = capsys.readouterr()
        assert "PHASE 2.7" in captured.out
        assert "Flag breakdown" in captured.out
