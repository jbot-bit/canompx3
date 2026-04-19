"""Tests for research/phase_2_8_multi_year_regime_stratification.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research import phase_2_8_multi_year_regime_stratification as mod


class TestCanonicalDelegation:
    def test_imports_compute_mode_a(self):
        from research.mode_a_revalidation_active_setups import compute_mode_a
        assert mod.compute_mode_a is compute_mode_a

    def test_stratify_years_covers_phase_2_7_vol_findings(self):
        # Phase 2.7 caveat (a) identified 2020, 2022, 2024 as elevated-ATR years
        assert mod.STRATIFY_YEARS == (2020, 2022, 2024)

    def test_thresholds_match_phase_2_7(self):
        assert mod.FLAG_DELTA_THRESHOLD == 0.03
        assert mod.FLAG_UNEVALUABLE_MIN_N == 30


class TestSubsetT:
    def test_standard(self):
        assert math.isclose(mod.subset_t(0.1, 1.0, 100), 1.0, abs_tol=1e-9)

    def test_edge_cases_nan(self):
        assert math.isnan(mod.subset_t(None, 1.0, 100))
        assert math.isnan(mod.subset_t(0.1, None, 100))
        assert math.isnan(mod.subset_t(0.1, 0.0, 100))


class TestClassifyPattern:
    def test_recurring_2_years(self):
        pattern = mod.classify_pattern(
            year_deltas={2020: 0.05, 2022: 0.04, 2024: 0.01},
            year_expr_pure={2020: -0.08, 2022: -0.06, 2024: 0.02},
            year_n={2020: 50, 2022: 60, 2024: 70},
        )
        assert pattern.startswith("RECURRING_VOL_DRAG")
        assert "2020" in pattern and "2022" in pattern

    def test_recurring_all_3_years(self):
        pattern = mod.classify_pattern(
            year_deltas={2020: 0.05, 2022: 0.04, 2024: 0.06},
            year_expr_pure={2020: -0.08, 2022: -0.06, 2024: -0.10},
            year_n={2020: 50, 2022: 60, 2024: 70},
        )
        assert pattern.startswith("RECURRING_VOL_DRAG")
        assert "2020" in pattern and "2022" in pattern and "2024" in pattern

    def test_single_year_drag(self):
        pattern = mod.classify_pattern(
            year_deltas={2020: 0.005, 2022: 0.01, 2024: 0.06},
            year_expr_pure={2020: 0.10, 2022: 0.08, 2024: -0.10},
            year_n={2020: 50, 2022: 60, 2024: 70},
        )
        assert pattern == "SINGLE_YEAR_DRAG (2024)"

    def test_vol_neutral(self):
        pattern = mod.classify_pattern(
            year_deltas={2020: 0.01, 2022: 0.005, 2024: 0.005},
            year_expr_pure={2020: 0.10, 2022: 0.08, 2024: 0.09},
            year_n={2020: 50, 2022: 60, 2024: 70},
        )
        assert pattern == "VOL_NEUTRAL"

    def test_unevaluable_all_thin(self):
        pattern = mod.classify_pattern(
            year_deltas={2020: 0.05, 2022: 0.04, 2024: 0.06},
            year_expr_pure={2020: -0.08, 2022: -0.06, 2024: -0.10},
            year_n={2020: 10, 2022: 5, 2024: 15},  # all < 30
        )
        assert pattern == "UNEVALUABLE"

    def test_thin_year_excluded_from_drag_count(self):
        # 2020 thin-N; only 2024 qualifies → SINGLE_YEAR_DRAG
        pattern = mod.classify_pattern(
            year_deltas={2020: 0.05, 2022: 0.005, 2024: 0.06},
            year_expr_pure={2020: -0.08, 2022: 0.08, 2024: -0.10},
            year_n={2020: 10, 2022: 60, 2024: 70},
        )
        assert pattern == "SINGLE_YEAR_DRAG (2024)"

    def test_requires_both_lift_and_drag_conditions(self):
        # 2020 has lift but year-expr not negative enough — NOT drag
        pattern = mod.classify_pattern(
            year_deltas={2020: 0.05, 2022: 0.01, 2024: 0.01},
            year_expr_pure={2020: 0.01, 2022: 0.08, 2024: 0.09},
            year_n={2020: 50, 2022: 60, 2024: 70},
        )
        assert pattern == "VOL_NEUTRAL"


class TestWindowStats:
    def test_invalid_session_raises(self):
        import duckdb
        con = duckdb.connect(":memory:")
        bad = {
            "instrument": "MNQ",
            "orb_label": "NOT_A_SESSION",
            "orb_minutes": 5,
            "entry_model": "E2",
            "confirm_bars": 1,
            "rr_target": 1.5,
            "execution_spec": '{"direction":"long"}',
        }
        with pytest.raises(ValueError, match="SESSION_CATALOG"):
            mod._window_stats(con, bad)
        con.close()


@pytest.mark.integration
class TestEndToEnd:
    def test_main_runs(self, tmp_path, monkeypatch, capsys):
        if not Path(mod.GOLD_DB_PATH).exists():
            pytest.skip("gold.db not available")
        monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path)
        rc = mod.main()
        assert rc == 0
        csv = tmp_path / "phase_2_8_multi_year_regime_stratification.csv"
        assert csv.exists()
        df = pd.read_csv(csv)
        assert 10 <= len(df) <= 200
        for y in mod.STRATIFY_YEARS:
            assert f"ex{y}_expr" in df.columns
            assert f"y{y}_expr" in df.columns
            assert f"delta{y}" in df.columns
        valid_patterns = {"VOL_NEUTRAL", "UNEVALUABLE", "ERROR"}
        # Plus prefix-patterned SINGLE_YEAR_DRAG (...) and RECURRING_VOL_DRAG (...)
        for pat in df["pattern"].unique():
            if pat in valid_patterns:
                continue
            assert pat.startswith("SINGLE_YEAR_DRAG") or pat.startswith("RECURRING_VOL_DRAG"), \
                f"unexpected pattern {pat!r}"
        captured = capsys.readouterr()
        assert "PHASE 2.8" in captured.out
